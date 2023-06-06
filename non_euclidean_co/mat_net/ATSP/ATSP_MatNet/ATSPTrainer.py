
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import wandb

import random
import math
import torch
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model
from ATSProblemDef import load_single_problem_from_file

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped



class ATSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 validate_params,
                 il_env_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.validate_params = validate_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        torch.manual_seed(self.trainer_params['seed'])
        random.seed(self.trainer_params['seed'])

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.il_env = Env(**il_env_params)

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        self.cum_samples = 0
        # Load all problems into tensor
        self.logger.info(" *** Loading Saved Problems *** ")
        saved_problem_folder = self.validate_params['saved_problem_folder']
        saved_problem_filename = self.validate_params['saved_problem_filename']
        file_count = self.validate_params['file_count']
        node_cnt = self.env_params['node_cnt']
        scaler = self.env_params['problem_gen_params']['scaler']
        self.all_problems = torch.empty(size=(file_count, node_cnt, node_cnt))
        for file_idx in range(file_count):
            formatted_filename = saved_problem_filename.format(file_idx)
            full_filename = os.path.join(saved_problem_folder, formatted_filename)
            problem = load_single_problem_from_file(full_filename, node_cnt, scaler)
            self.all_problems[file_idx] = problem
        self.logger.info("Done. ")

    def run(self):
        if self.trainer_params['wandb']:
            wandb.init(project='MatNet Sample Efficiency', 
                        entity='hyeonah_kim',
                        group='atsp' + str(self.env_params['node_cnt']),
                        name=self.trainer_params['logging']['run_name'] + str(self.result_folder.split('_')[1]),
                        reinit=True,
                        config=self.trainer_params)

        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            if epoch > 1:
                self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            no_aug_mean, aug_mean = self._validate()

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size
            self.cum_samples += (batch_size * self.env_params['pomo_size'])

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))
            
        # Log Once, for each epoch
        self.logger.info("{} samples are used.".format(self.cum_samples))
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~)
        action_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state, fixed_start=self.trainer_params['fixed_start'])
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            action_list = torch.cat((action_list, selected[:, :, None]), dim=2)

        # Loss
        ###############################################
        if self.trainer_params['instance_shared']:
            advantage = (reward - reward.float().mean())/(reward.float().std() + 1e-8)
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        # max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        grad_norms = clip_grad_norms(self.optimizer.param_groups, 1.0)
        self.optimizer.step()

        if self.trainer_params['il_coefficient'] > 0.:
            greedy_actions = self._rollout_for_self_training()
                
            il_prob_list = torch.zeros(size=(batch_size, self.il_env.pomo_size, 0))

            if self.trainer_params['no_symmetric']:
                actions = greedy_actions
            else:
                actions = self._symmetric_action(greedy_actions)

            for _ in range(self.trainer_params['distil_n_iter']):
                # POMO Rollout
                ###############################################
                reset_state, _, _ = self.il_env.reset()
                self.model.pre_forward(reset_state)
                state, reward, done = self.il_env.pre_step()
                for step in range(actions.shape[-1]):
                    a = actions[:, :, step].long()
                    # print(a)
                    selected, prob = self.model(state, a, fixed_start=self.trainer_params['fixed_start'])
                    # shape: (batch, pomo)
                    state, reward, done = self.il_env.step(selected)

                    il_prob_list = torch.cat((il_prob_list, prob[:, :, None]), dim=2)
                    # action_list = torch.cat((action_list, selected[:, :, None]), dim=2)

                sub_len = random.randint(1, 10)
                # sub_len = 1
                log_likelihood_IL = il_prob_list[:, 0, sub_len:].log().sum(dim=-1)

                il_loss = (-1) * self.trainer_params['il_coefficient'] * log_likelihood_IL.mean()

                # Step & Return
                ###############################################
                self.model.zero_grad()
                il_loss.backward()
                if self.trainer_params['clipping'] > 0. :
                    grad_norms = clip_grad_norms(self.optimizer.param_groups, self.trainer_params['clipping'])
                self.optimizer.step()

                if not self.trainer_params['no_symmetric']:
                    actions = self._symmetric_action(greedy_actions)
                il_prob_list = torch.zeros(size=(batch_size, self.il_env.pomo_size, 0))

        # Logging
        if self.trainer_params['wandb']:
            log_dict = {'avg_score': score_mean.item(),
                        'rl_loss': loss_mean.item(),
                        'il_loss': il_loss.item() if self.trainer_params['il_coefficient'] > 0. else 0,
                        'cum_samples': self.cum_samples,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }

            wandb.log(log_dict)
        
        return score_mean.item(), loss_mean.item()
    
    def _validate(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.validate_params['file_count']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.validate_params['test_batch_size'], remaining)

            score, aug_score = self._validate_one_batch(episode, episode+batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" {}".format(self.trainer_params['logging']['run_name']))
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                # self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

                if self.trainer_params['wandb']:
                    log_dict = {'val_avg_score': score_AM.avg,
                                'val_aug_avg_score': aug_score_AM.avg,
                                'cum_samples': self.cum_samples
                            }

                    wandb.log(log_dict)

        return score_AM.avg, aug_score_AM.avg


    def _validate_one_batch(self, idx_start, idx_end):

        batch_size = idx_end-idx_start
        problems_batched = self.all_problems[idx_start:idx_end]

        # Augmentation
        ###############################################
        if self.validate_params['augmentation_enable']:
            aug_factor = self.validate_params['aug_factor']

            batch_size = aug_factor*batch_size
            problems_batched = problems_batched.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_batched, scaler=self.validate_params['scaler'])
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state, fixed_start=self.trainer_params['fixed_start'])
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

            # Return
            ###############################################
            batch_size = batch_size//aug_factor
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            # Rescale
            # aug_reward /= float(self.validate_params['scaler'])

            # max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -aug_reward[0, :, 0].float().mean()  # negative sign to make positive value

            # max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -aug_reward[0, :, 0].float().mean()  # negative sign to make positive value

            return no_aug_score.item(), aug_score.item()


    def _symmetric_action(self, action):
        batch_size, sample_size, action_len = action.shape
            
        permuted_indice = torch.arange(action_len).repeat(batch_size, sample_size, 1)
        start = torch.randint(action_len - 1, size=(batch_size, sample_size, 1)) + 1
        random_permuted = (permuted_indice + start) % action_len
        sym_action = torch.gather(action, dim=-1, index=random_permuted)

        # return sym_action
        return action

    def _rollout_for_self_training(self):
        batch_size = self.trainer_params['train_batch_size']
        confidence_threshold = self.trainer_params['confidence_threshold']

        # greedy 
        self.model.eval()
        prob_list = torch.zeros(size=(batch_size, self.il_env.pomo_size, 0))
        greedy_action_list = torch.zeros(size=(batch_size, self.il_env.pomo_size, 0))

        with torch.no_grad():
            self.il_env.load_problems_manual(self.env.problems)
            reset_state, _, _ = self.il_env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            state, reward, done = self.il_env.pre_step()
            while not done:
                selected, prob = self.model(state, fixed_start=self.trainer_params['fixed_start'])
                # shape: (batch, pomo)
                state, reward, done = self.il_env.step(selected)

                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
                greedy_action_list = torch.cat((greedy_action_list, selected[:, :, None]), dim=2)
        
        if confidence_threshold < 1.0:
            ll = prob_list.squeeze(1).log().sum(dim=-1)
            confident_indices = ll.topk(k=int(batch_size * confidence_threshold), sorted=True).indices

            # Select corresponding x values
            x = self.il_env.problems[confident_indices, :, :].repeat(int(1 / confidence_threshold), 1, 1)
            pi = greedy_action_list[confident_indices, :, :].repeat(int(1 / confidence_threshold), 1, 1)

            self.il_env.load_problems_manual(x)
        else:
            pi = greedy_action_list
        
        self.model.train()
        return pi
