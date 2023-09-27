import os
import time
import math
import random

import wandb
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from symrd_utils import rollout_for_self_distillation, symmetric_action


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    
    ############################### [SymRD] ###################################
    random.seed(opts.seed)

    if opts.baseline == 'rollout':
        cum_samples = epoch * (2*opts.epoch_size + opts.val_size)  # rollout baseline 
        # if epoch > 0:
        #     cum_samples += opts.epoch_size 
    else:
        cum_samples = epoch * opts.epoch_size
    ###########################################################################
    
    start_time = time.time()

    # if not opts.no_tensorboard:
    #     tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        ############################### [SymRD] ###################################
        if opts.baseline is None or opts.baseline in ['critic', 'no_baseline']:
            cum_samples += opts.batch_size  # batch.shape[0]
        elif opts.baseline == 'rollout':
            if epoch == 0:
                cum_samples += opts.batch_size * 2 
            else:
                cum_samples += opts.batch_size  # batch['data'].shape[0]
        ###########################################################################


        if opts.method == 'gfn':
            train_batch_gfn(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                cum_samples,
                opts
            )
        elif opts.method == 'ppo':
            train_batch_ppo(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                cum_samples,
                opts
            )
        else:
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                cum_samples,
                opts
            )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if opts.wandb != 'disabled':
        wandb_log_dict = {'val_avg_cost': avg_reward,
                          'cum_samples': cum_samples,
                          'step': step}

        wandb.log(wandb_log_dict)

    # if not opts.no_tensorboard:
    #     print("{} samples are used.".format(cum_samples))
    #     tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def distill_model(model, optimizer, opts, x, pi=None, ll_clip=0.):
    if pi is None:
        new_pi = rollout_for_self_distillation(model, opts, x)
    else:
        new_pi = pi.clone()
    
    for _ in range(opts.distil_loop):
        sub_len = random.randint(1, 10)

        if opts.no_symmetric:
            _, _, _, log_likelihood_IL, _ = model(x, action=new_pi, sub_len=opts.graph_size - sub_len)
        else:
            if opts.problem == 'tsp':
                new_x = x.repeat(opts.sym_width, 1, 1)
                action = symmetric_action(new_pi.repeat(opts.sym_width, 1), opts)

            elif opts.problem == 'cvrp':
                new_x = {
                    'loc': x['loc'].repeat(opts.sym_width, 1, 1),
                    'demand': x['demand'].repeat(opts.sym_width, 1),
                    'depot': x['depot'].repeat(opts.sym_width, 1)
                }

                new_x = x.repeat(opts.sym_width, 1, 1)
                action = symmetric_action(new_pi.repeat(opts.sym_width, 1), opts)
            
            _, _, _, log_likelihood_IL, _ = model(new_x, action=action, sub_len=opts.graph_size - sub_len)

        if ll_clip > 0:
            log_likelihood_IL = torch.clamp(log_likelihood_IL, -ll_clip, ll_clip)

        il_loss = (-1) * opts.il_coefficient * log_likelihood_IL.mean()
        
        optimizer.zero_grad()
        il_loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()

    return il_loss


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        cum_samples,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)
    # cost, log_likelihood, entropy = model(x, return_entropy=True)  # MaxEnt AM

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # reinforce_loss = ((cost - bl_val - 0.01 * entropy.sum(-1).clone().detach()) * log_likelihood).mean()  # MaxEnt AM
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    ############################### [SymRD] ###################################
    if opts.il_coefficient > 0 and (step + 1) % opts.distil_every == 0:
        distill_loss = distill_model(model, optimizer, opts, x)
    else:
        distill_loss = 0

    wandb_log_dict = {'avg_cost': cost.mean().item(),
                    'cum_samples': cum_samples,
                    'rl_loss': reinforce_loss.item(),
                    'il_loss': distill_loss, 
                    'nll': -log_likelihood.mean().item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': step}

    # Logging
    if opts.wandb != 'disabled':
        wandb.log(wandb_log_dict)


def train_batch_ppo(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        cum_samples,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Exploration
    with torch.no_grad():
        cost, log_likelihood_old,pi = model(x,return_pi = True)
    # cost, log_likelihood, entropy = model(x, return_entropy=True)  # MaxEnt AM

    # Evaluate baseline, get baseline loss if any (only for critic)


    for i in range(opts.k_step):
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        _, log_likelihood = model(x,action=pi)
        # Perform backward pass and optimization step
        
        advantage = bl_val - cost
        ratio = log_likelihood - log_likelihood_old

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-opts.eps_clip, 1+opts.eps_clip) * advantage
        
        reinforce_loss = -torch.min(surr1,surr2).mean()
        loss = reinforce_loss + bl_loss
        
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()

    ############################### [SymRD] ###################################
    if opts.il_coefficient > 0 and (step + 1) % opts.distil_every == 0:
        distill_loss = distill_model(model, optimizer, opts, x, ll_clip=ratio.mean().item())
    else:
        distill_loss = 0

    wandb_log_dict = {'avg_cost': cost.mean().item(),
                    'cum_samples': cum_samples,
                    'rl_loss': reinforce_loss.item(),
                    'il_loss': distill_loss, 
                    'nll': -log_likelihood.mean().item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': step}

    # Logging
    if opts.wandb != 'disabled':
        wandb.log(wandb_log_dict)


def train_batch_gfn(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        cum_samples,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    
    # Exploration 
    with torch.no_grad():
        cost, _, pi = model(x, return_pi=True)

    # Training
    for i in range(opts.k_step):
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        _, log_likelihood,logZ = model(x,action=pi,gfn=True)

        forward_flow = log_likelihood + logZ.view(-1)
        backward_flow = math.log(1/(2*pi.shape[0])) - (cost-bl_val)*opts.beta
        
        reinforce_loss = torch.pow(forward_flow-backward_flow, 2).mean()
        loss = reinforce_loss + bl_loss

        optimizer.zero_grad()
        loss.backward()

        # Clip gradient norms and get (clipped) gradient norms for logging
        grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        optimizer.step()

    ############################### [SymRD] ###################################
    if opts.il_coefficient > 0 and (step + 1) % opts.distil_every == 0:
        distill_loss = distill_model(model, optimizer, opts, x)
    else:
        distill_loss = 0

    wandb_log_dict = {'avg_cost': cost.mean().item(),
                    'cum_samples': cum_samples,
                    'rl_loss': reinforce_loss.item(),
                    'il_loss': distill_loss, 
                    'nll': -log_likelihood.mean().item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': step}

    # Logging
    if opts.wandb != 'disabled':
        wandb.log(wandb_log_dict)
