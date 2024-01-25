import torch
import random

from nets.attention_model import set_decode_type


def get_sym_actions_and_log_probs(action, opts, x=None, model=None, sub_len=0, beta=1.0):
    batch_size, action_len = action.shape
    assert opts.problem == 'tsp' and opts.transform_opt == 'weighted'
    
    # k-cyclic permutation
    permuted_indice = torch.arange(action_len).repeat(batch_size, 1).to(x.device)
    
    new_x = x.repeat(100, 1, 1)
    pi = action.repeat(100, 1)

    start = torch.arange(100).repeat_interleave(batch_size).to(x.device)
    permuted = (permuted_indice.repeat(100, 1) + start.view(-1, 1)) % action_len
    permuted = torch.gather(pi, dim=-1, index=permuted.to(x.device))
    permuted[50*batch_size:, :] = permuted[50*batch_size:, :].flip(dims=[-1])

    with torch.no_grad():
        _, sym_ll = model(new_x, action=permuted, sub_len=0)

    permuted = permuted.reshape(100, -1, action_len)
    sym_ll = sym_ll.reshape(100, -1).detach()
    min_ll, _ = sym_ll.min(dim=0)
    sum_ll = sym_ll.sum(dim=0)

    ll_diff = (sym_ll.mean(dim=0) - min_ll).mean()

    new_x = x.repeat(10, 1, 1)
    pi = action.repeat(10, 1)

    # start = torch.arange(10).repeat_interleave(batch_size).to(x.device)
    start = torch.randint(action_len - 1, size=(batch_size, 10)).to(x.device) + 1
    permuted = (permuted_indice.repeat(10, 1) + start.view(-1, 1)) % action_len
    permuted = torch.gather(pi, dim=-1, index=permuted.to(x.device))
    # permuted[50*batch_size:, :] = permuted[50*batch_size:, :].flip(dims=[-1])

    # with torch.no_grad():
    #     _, sym_ll = model(new_x, action=permuted, sub_len=0)
    # print(new_x.shape, permuted.shape, sub_len)
    _, sym_ll, sub_ll1, log_likelihood_IL, _ = model(new_x, action=permuted, sub_len=opts.graph_size - sub_len)
    # print(sym_ll.shape, sub_ll1.shape, log_likelihood_IL.shape)

    sym_ll = (sub_ll1 + log_likelihood_IL).reshape(10, -1).detach()
    min_ll, _ = sym_ll.min(dim=0)
    sum_ll = sym_ll.sum(dim=0)

    weights = ((-1) * beta * sym_ll).exp() / (((-1) * beta * sum_ll).exp() + 1e-8)
    # print(weights.min(dim=1), weights.sum(dim=1))
    # print(sym_ll[0, 0])

    # return permuted.reshape(100, -1, action_len), sym_ll.reshape(100, -1)
    return ll_diff, weights, log_likelihood_IL.reshape(10, -1)


def symmetric_action(action, opts, x=None, model=None, return_ll=False):
    batch_size, action_len = action.shape
    
    if opts.problem == 'tsp':
        # k-cyclic permutation
        permuted_indice = torch.arange(action_len).repeat(batch_size, 1).to(opts.device)
        
        if opts.transform_opt == 'identical':
            sym_action = action.flip(dims=[-1])
            # start = torch.ones(size=(batch_size, 1)).int() + 1
        elif opts.transform_opt in ['adv_all', 'adv_sample']:
            new_x = x.repeat(100, 1, 1)
            pi = action.repeat(100, 1)

            start = torch.arange(100).repeat_interleave(batch_size).to(opts.device)
            permuted = (permuted_indice.repeat(100, 1) + start.view(-1, 1)) % action_len
            permuted = torch.gather(pi, dim=-1, index=permuted.to(opts.device))
            permuted[50*batch_size:, :] = permuted[50*batch_size:, :].flip(dims=[-1])

            with torch.no_grad():
                _, sym_ll = model(new_x, action=permuted, sub_len=0)

            sym_ll = sym_ll.reshape(100, -1)
            min_ll, idx = sym_ll.min(dim=0)

            if opts.transform_opt == 'adv_sample':
                # print((-1) * opts.inverse_temp * sym_ll[0, 0])
                exp = (torch.clamp((-1) * opts.inverse_temp * sym_ll, 0, 500)).double().exp()
                # print(nll.min(dim=0), nll.max(dim=0), nll.sum(dim=0))
                weights = exp / (exp.sum(dim=0) + 1e-8)
                # print(weights.min(dim=0), weights.max(dim=0), weights.sum(dim=0))
                # prob = ((-sym_ll[:, 0, :]).exp() / (-log_p1[:, 0, :]).exp().sum(dim=1)[:, None])
                idx = torch.multinomial(weights, 1).view(-1)
            
            permuted = permuted.reshape(100, -1, action_len)
            sym_action = permuted[idx, torch.arange(permuted.size(1))]

            return sym_action, (sym_ll.mean(dim=0) - min_ll).mean()
        else:

            start = torch.randint(action_len - 1, size=(batch_size, 1)).to(x.device) + 1
        
            random_permuted = (permuted_indice + start) % action_len
            permuted = torch.gather(action, dim=-1, index=random_permuted.to(opts.device))
            
            if torch.rand(1) >= 0.5:
                sym_action = permuted
            else:  # flipping
                sym_action = permuted.flip(dims=[-1])

    elif opts.problem == 'cvrp':
        # action does not start with 0
        if opts.transform_opt == 'identical':
            flipped = action.flip(dims=[-1])
            start = (flipped > 0).long().argmax(dim=1)
            flipped_idx = (torch.arange(action_len).repeat(batch_size, 1).to(opts.device) + start.view(-1, 1)) % action_len
            sym_action = torch.gather(flipped, dim=-1, index=flipped_idx)
        else:
            zero_idx = torch.nonzero(action == 0).view(batch_size, -1, len(action.shape))
            tour_len = zero_idx[:, :, 1] - torch.cat([(-1)*torch.ones(batch_size, 1, dtype=zero_idx.dtype).to(opts.device), zero_idx[:, :-1, 1]], dim=1)-1

            permuted_list = []
            for b in range(batch_size):
                chunk = torch.split(action[b], (tour_len[b] + 1).tolist())
                valid_chunk = []
                for c in chunk:
                    if len(c) > 1:
                        if torch.rand(1) >= 0.5:
                            valid_chunk.append(c)
                        else:
                            flipped = torch.cat([c[:-1].flip(dims=[-1]), torch.tensor([0]).to(c.device)], dim=0)
                            valid_chunk.append(flipped)
                # valid_chunk = [c if torch.rand(1) >= 0.5 else c.flip(dims=[-1]) for c in chunk if len(c) > 1]
                tmp = torch.cat(random.sample(valid_chunk, k=len(valid_chunk)), dim=0)
                permuted_list.append(torch.cat([tmp, torch.zeros(action_len - tmp.shape[-1], dtype=tmp.dtype).to(opts.device)], dim=0))

            permuted = torch.stack(permuted_list)

            if torch.rand(1) >= .5:
                sym_action = permuted
            else:
                flipped = permuted.flip(dims=[-1])
                start = (flipped > 0).long().argmax(dim=1)
                flipped_idx = (torch.arange(action_len).repeat(batch_size, 1).to(opts.device) + start.view(-1, 1)) % action_len
                sym_action = torch.gather(flipped, dim=-1, index=flipped_idx)

    return sym_action


def calc_ll_difference(action, x, model):
    # to measure differences of symmetric likelihoods
    batch_size, action_len = action.shape
    new_x = x.repeat(100, 1, 1)
    pi = action.repeat(100, 1)

    permuted_indice = torch.arange(action_len).repeat(batch_size, 1).to(x.device)

    start = torch.arange(100).repeat_interleave(batch_size).to(x.device)
    permuted = (permuted_indice.repeat(100, 1) + start.view(-1, 1)) % action_len
    permuted = torch.gather(pi, dim=-1, index=permuted.to(x.device))
    permuted[50*batch_size:, :] = permuted[50*batch_size:, :].flip(dims=[-1])

    with torch.no_grad():
        _, sym_ll = model(new_x, action=permuted, sub_len=0)
    
    min_ll, _ = sym_ll.reshape(100, -1).min(dim=0)

    return (sym_ll.mean(dim=0) - min_ll).mean()


def rollout_for_self_distillation(model, opts, x):

    model.eval()
    set_decode_type(model, "greedy")
    with torch.no_grad():
        _, log_likelihood, pi = model(x, return_pi=True)
            
    model.train()
    set_decode_type(model, "sampling")

    return pi
