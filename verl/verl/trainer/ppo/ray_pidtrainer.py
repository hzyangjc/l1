import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos, ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, WorkerType, Role, ResourcePoolManager
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.trainer.pid_lagrange.pid_lag import PIDLagrangian
from verl.trainer.ppo.ray_trainer import dataprotoitem_to_dataproto, reduce_metrics, compute_data_metrics, compute_timing_metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
        # prepare response group
        # TODO: add other ways to estimate advantages
        if adv_estimator == 'gae':
            values = data.batch['values']
            responses = data.batch['responses']
            response_length = responses.size(-1)
            attention_mask = data.batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            token_level_rewards = data.batch['token_level_rewards']
            advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                        values=values,
                                                                        eos_mask=response_mask,
                                                                        gamma=gamma,
                                                                        lam=lam)
            data.batch['advantages'] = advantages
            data.batch['returns'] = returns
        elif adv_estimator == 'grpo':
            token_level_rewards = data.batch['token_level_rewards']
            token_level_costs = data.batch['token_level_costs']
            index = data.non_tensor_batch['uid']
            responses = data.batch['responses']
            response_length = responses.size(-1)
            attention_mask = data.batch['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            advantages_c, returns_c = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_costs,
                                                                            eos_mask=response_mask,
                                                                            index=index)
            data.batch['advantages'] = advantages
            data.batch['returns'] = returns
            data.batch['advantages_c'] = advantages_c
            data.batch['returns_c'] = returns_c
        else:
            raise NotImplementedError
        return data

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last
        
    
class RayPIDTrainer(RayPPOTrainer):
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):
        super().__init__(config, tokenizer, role_worker_mapping, resource_pool_manager, ray_worker_group_cls, reward_fn, val_reward_fn)
        self._lagrange: PIDLagrangian = PIDLagrangian(**self.config.lagrange_cfgs)

    def fit(self):

        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # perform validation before training  
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for _ in range(self.config.trainer.total_epochs):  
            
            for batch_dict in self.train_dataloader: 
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])  

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        # breakpoint()
                        print(f'gen_batch: {gen_batch}')
                        
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)  
                    batch = batch.union(gen_batch_output)  

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)  
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor, cost_tensor = self.rm_wg.compute_rm_score(batch)  
                            batch = batch.union(reward_tensor, cost_tensor)
                        
                        # TODO: 实现cost_tensor的计算
                        # 适配只返回一个值的reward_fn函数
                        reward_result = self.reward_fn(batch)
                        if isinstance(reward_result, tuple) and len(reward_result) == 2:
                            reward_tensor, cost_tensor = reward_result
                        else:
                            # 如果只返回一个值，将reward作为奖励，成本设为0
                            reward_tensor = reward_result
                            # 创建一个与reward_tensor形状相同但全为0的tensor作为cost_tensor
                            cost_tensor = torch.zeros_like(reward_tensor)
                            
                        batch.batch['token_level_scores'] = reward_tensor    
                        batch.batch['token_level_costs'] = cost_tensor

                        # Rejection sampling based on rewards  
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        solve_mean = 0
                        solve_perfect = 0
                        solve_perfect_all = 0
                        token_score = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards > 0).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                            solve_mean += sum(uid_rewards > 0)/len(uid_rewards)
                            solve_perfect += sum(uid_rewards > 0.9)/len(uid_rewards)
                            solve_perfect_all += 1 if (uid_rewards > 0.9).all() else 0
                            # If less than 0, then 1-x, else x
                            token_score += sum(1 + x if x < 0 else x for x in uid_rewards)/len(uid_rewards)
                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all
                        metrics['batch/solve_mean'] = solve_mean/len(unique_uids)
                        metrics['batch/solve_perfect'] = solve_perfect/len(unique_uids)
                        metrics['batch/solve_perfect_all'] = solve_perfect_all/len(unique_uids)
                        metrics['batch/token_score'] = token_score/len(unique_uids)
                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch_item = batch[valid_mask]
                            batch = dataprotoitem_to_dataproto(batch_item)
                            # Round down to the nearest multiple of world size
                            num_trainer_replicas = self.actor_rollout_wg.world_size 
                            max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                            if not max_batch_size:
                                # give up, you got everything either all wrong or right.
                                continue

                            size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                            size_mask[:max_batch_size] = True
                            batch_item = batch[size_mask]
                            batch = dataprotoitem_to_dataproto(batch_item)

                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        

                        # compute advantages, executed on the driver process  
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # TODO:update Lagrangian multiplier
                    Jc = batch.batch['token_level_costs']   # Jc is the discount sum cost of the current episode，
                    cost_avg = torch.mean(Jc).item()  # 计算平均成本
                    self._lagrange.pid_update(cost_avg)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor  
                        with _timer('update_actor', timing_raw):
                            # TODO: 修改fsdp_workers中的update_actor参数
                            batch.meta_info['lagrangian_multiplier'] = self._lagrange.lagrangian_multiplier
                            actor_output = self.actor_rollout_wg.update_actor(batch)  
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
