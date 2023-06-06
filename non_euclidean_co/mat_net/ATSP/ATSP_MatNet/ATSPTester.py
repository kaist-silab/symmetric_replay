
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

import torch

import os
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model

from utils.utils import get_result_folder, AverageMeter, TimeEstimator

from ATSProblemDef import load_single_problem_from_file


class ATSPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        torch.manual_seed(1234)

        # Restore
        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        # Load all problems into tensor
        self.logger.info(" *** Loading Saved Problems *** ")
        saved_problem_folder = self.tester_params['saved_problem_folder']
        saved_problem_filename = self.tester_params['saved_problem_filename']
        file_count = self.tester_params['file_count']
        node_cnt = self.env_params['node_cnt']
        scaler = self.env_params['problem_gen_params']['scaler']
        self.all_problems = torch.empty(size=(file_count, node_cnt, node_cnt))
        for file_idx in range(file_count):
            formatted_filename = saved_problem_filename.format(file_idx)
            full_filename = os.path.join(saved_problem_folder, formatted_filename)
            problem = load_single_problem_from_file(full_filename, node_cnt, scaler)
            self.all_problems[file_idx] = problem
        self.logger.info("Done. ")

    # def run(self):

    #     self.time_estimator.reset()

    #     score_AM = AverageMeter()
    #     aug_score_AM = AverageMeter()

    #     test_num_episode = self.tester_params['file_count']
    #     episode = 0

    #     while episode < test_num_episode:

    #         remaining = test_num_episode - episode
    #         batch_size = min(self.tester_params['test_batch_size'], remaining)

    #         score, aug_score = self._test_one_batch(episode, episode+batch_size)

    #         score_AM.update(score, batch_size)
    #         aug_score_AM.update(aug_score, batch_size)

    #         episode += batch_size

    #         ############################
    #         # Logs
    #         ############################
    #         elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
    #         self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
    #             episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

    #         all_done = (episode == test_num_episode)

    #         if all_done:
    #             self.logger.info(" *** Test Done *** ")
    #             self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
    #             # self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    # def _test_one_batch(self, idx_start, idx_end):

    #     batch_size = idx_end-idx_start
    #     problems_batched = self.all_problems[idx_start:idx_end]

    #     # Augmentation
    #     ###############################################
    #     if self.tester_params['augmentation_enable']:
    #         aug_factor = self.tester_params['aug_factor']

    #         batch_size = aug_factor*batch_size
    #         problems_batched = problems_batched.repeat(aug_factor, 1, 1)
    #     else:
    #         aug_factor = 1

    #     # Ready
    #     ###############################################
    #     self.model.eval()
    #     with torch.no_grad():
    #         self.env.load_problems_manual(problems_batched)
    #         reset_state, _, _ = self.env.reset()
    #         self.model.pre_forward(reset_state)

    #         # POMO Rollout
    #         ###############################################
    #         state, reward, done = self.env.pre_step()
    #         while not done:
    #             selected, _ = self.model(state)
    #             # shape: (batch, pomo)
    #             state, reward, done = self.env.step(selected)

    #         # Return
    #         ###############################################
    #         batch_size = batch_size//aug_factor
    #         aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
    #         # shape: (augmentation, batch, pomo)

    #         max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
    #         # shape: (augmentation, batch)
    #         no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

    #         max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
    #         # shape: (batch,)
    #         aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

    #         return no_aug_score.item(), aug_score.item()

    def run(self):
        self.time_estimator.reset()

        aug_score_AM = AverageMeter()
        score_AM = AverageMeter()
        prob_AM = AverageMeter()

        test_num_episode = self.tester_params['file_count']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            aug_score, score, prob_loss = self._validate_one_batch(episode, episode+batch_size)

            aug_score_AM.update(aug_score, batch_size)
            score_AM.update(score, batch_size)
            prob_AM.update(prob_loss, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], aug_score:{:.3f}, score:{:.3f}, prob_loss:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, aug_score, score, prob_loss))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                # self.logger.info(" {}".format(self.trainer_params['logging']['run_name']))
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                self.logger.info(" SYMMETRIC PROB LOSS: {:.4f} ".format(prob_AM.avg))

        return score_AM.avg, prob_AM.avg

    def _validate_one_batch(self, idx_start, idx_end):

        batch_size = idx_end-idx_start
        problems_batched = self.all_problems[idx_start:idx_end]

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']

            batch_size = aug_factor*batch_size
            problems_batched = problems_batched.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(problems_batched, scaler=self.tester_params['scaler'])
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            ###############################################
            prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
            greedy_action_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
            
            state, reward, done = self.env.pre_step()
            while not done:
                selected, prob = self.model(state, fixed_start=self.tester_params['fixed_start'])
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
                greedy_action_list = torch.cat((greedy_action_list, selected[:, :, None]), dim=2)


            # Symmetric Prob
            ###############################################
            sym_prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
            actions = self._symmetric_action(greedy_action_list)
            
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, sym_reward, done = self.env.pre_step()
            for step in range(actions.shape[-1]):
                a = actions[:, :, step].long()
                # print(a)
                selected, sym_prob = self.model(state, a, fixed_start=False)
                # shape: (batch, pomo)
                state, sym_reward, done = self.env.step(selected)

                sym_prob_list = torch.cat((sym_prob_list, sym_prob[:, :, None]), dim=2)
            
            # Return
            ###############################################
            batch_size = batch_size//aug_factor
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            # Rescale
            # aug_reward /= float(self.validate_params['scaler'])

            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            # no_aug_score = -aug_reward[0, :, 0].float().mean()  # negative sign to make positive value
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            # shape: (batch,)
            # aug_score = -aug_reward[0, :, 0].float().mean()  # negative sign to make positive value
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

            log_prob = prob_list.log().sum(dim=-1).view(-1)
            log_sym_prob = sym_prob_list.log().sum(dim=-1).view(-1)

            # print(prob_list.shape, log_prob.shape, sym_prob_list.shape, log_sym_prob.shape)
            # print(log_prob.mean(), log_sym_prob.mean())

            prob_l1_loss = torch.nn.functional.l1_loss(log_prob, log_sym_prob)

            return aug_score.item(), no_aug_score.item(), prob_l1_loss.item()


    def _symmetric_action(self, action):
        batch_size, sample_size, action_len = action.shape
            
        permuted_indice = torch.arange(action_len).repeat(batch_size, sample_size, 1)
        start = torch.randint(action_len - 1, size=(batch_size, sample_size, 1)) + 1
        random_permuted = (permuted_indice + start) % action_len
        sym_action = torch.gather(action, dim=-1, index=random_permuted)

        # print(action[0])
        # print(sym_action[0])
        return sym_action
