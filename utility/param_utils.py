import copy
import errno
import signal
import threading
import types
from contextlib import contextmanager

import functools

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def to(tensors, device, non_blocking=False):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to(t, device, non_blocking=non_blocking))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to(v, device, non_blocking=non_blocking)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            return tensors.to(device, non_blocking=non_blocking)
        else:
            return tensors


def zero_and_update(optimizers, loss):
    if isinstance(optimizers, (list, tuple)):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            ret = opt.step()
    else:
        optimizers.zero_grad(set_to_none=True)
        loss.backward()
        ret = optimizers.step()
    return ret

def put_theta(model, theta, data=False):
    if theta is None:
        return model

    def k_param_fn(tmp_model, name=None):
        if len(theta) == 0:
            return

        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name == '':
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))

        # WARN : running_mean, 和 running_var 不是 parameter，所以在 new 中不会被更新
        for (k, v) in tmp_model._parameters.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                if data:
                    tmp_model._parameters[k].data = theta[str(name + '.' + k)].clone().data
                else:
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]
            # else:
            #     print(name+'.'+k)
            # theta.pop(str(name + '.' + k))

        for (k, v) in tmp_model._buffers.items():
            if isinstance(v, torch.Tensor) and str(name + '.' + k) in theta.keys():
                if data:
                    tmp_model._buffers[k].data = theta[str(name + '.' + k)].data
                else:
                    tmp_model._buffers[k] = theta[str(name + '.' + k)]
            # else:
            #     print(k)
            # theta.pop(str(name + '.' + k))

    k_param_fn(model, name='')
    return model


def get_parameters(model):
    # note : you can direct manipulate these data reference which is related to the original models
    parameters = dict(model.named_parameters())
    states = dict(model.named_buffers())
    return parameters, states


def put_parameters(model, param, state, data=False):
    model = put_theta(model, param, data)
    model = put_theta(model, state, data)
    return model


def update_parameters(loss, names_weights_dict, lr, use_second_order, retain_graph=True, grads=None, ignore_keys=None):
    def contains(key, target_keys):
        if isinstance(target_keys, (tuple, list)):
            for k in target_keys:
                if k in key:
                    return True
        else:
            return key in target_keys

    new_dict = {}
    for name, p in names_weights_dict.items():
        if p.requires_grad:
            new_dict[name] = p
        # else:
        #     print(name)
    names_weights_dict = new_dict

    if grads is None:
        grads = torch.autograd.grad(loss, names_weights_dict.values(), create_graph=use_second_order, retain_graph=retain_graph, allow_unused=True)
    names_grads_wrt_params_dict = dict(zip(names_weights_dict.keys(), grads))
    updated_names_weights_dict = dict()

    for key in names_grads_wrt_params_dict.keys():
        if names_grads_wrt_params_dict[key] is None:
            continue  # keep the original state unchanged

        if ignore_keys is not None and contains(key, ignore_keys):
            # print(f'ignore {key}' )
            continue

        updated_names_weights_dict[key] = names_weights_dict[key] - lr * names_grads_wrt_params_dict[key]
    return updated_names_weights_dict


def cat_meta_data(data_list):
    new_data = {}
    for k in data_list[0].keys():
        l = []
        for data in data_list:
            l.append(data[k])
        new_data[k] = torch.cat(l, 0)
    return new_data


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# @timeout(3)
def get_image_and_label(loaders, idx_list, device):
    if not isinstance(idx_list, (list, tuple)):
        idx_list = [idx_list]

    x_list = []
    y_list = []
    for i in idx_list:
        # with Timer('load data from domain {}'.format(i), thresh=0.5):
        data = loaders[i].next()
        x, y = to(data, device)  # , non_blocking=True)
        # data = loaders[i].next()
        x_list.append(x)
        y_list.append(y)
    return torch.cat(x_list), torch.cat(y_list)


def split_image_and_label(data, size, loo=False):
    n_domains = list(data.values())[0].shape[0] // size
    idx_sequence = list(np.random.permutation(n_domains))
    if loo:
        n_domains = 2
    res = [{} for _ in range(n_domains)]

    for k, v in data.items():
        split_data = torch.split(v, size)
        if loo:  # meta_train, meta_test
            res[0][k] = torch.cat([split_data[_] for _ in idx_sequence[:len(split_data) - 1]])
            res[1][k] = split_data[idx_sequence[-1]]
        else:
            for i, d in enumerate(split_data):
                res[i][k] = d
    return res


def pairs_of_data(train_data, device, paths, random=False, same=False):
    if same:
        for i in range(paths):
            yield [get_image_and_label(train_data, [i], device), get_image_and_label(train_data, [i], device)]
        return

    if not random:
        idx_sequence = list(np.random.permutation(len(train_data)))
        for i in range(paths):
            meta_train_idx = copy.deepcopy(idx_sequence)
            meta_train_idx.pop(i)
            meta_test_idx = [idx_sequence[i]]
            yield [get_image_and_label(train_data, meta_train_idx, device), get_image_and_label(train_data, meta_test_idx, device)]
    else:
        for i in range(paths):
            idx_sequence = list(np.random.permutation(len(train_data)))
            meta_train_idx = idx_sequence[:2]
            meta_test_idx = idx_sequence[2:]
            yield [get_image_and_label(train_data, meta_train_idx, device), get_image_and_label(train_data, meta_test_idx, device)]


def split_data(train_data, device, leave_one_out=False):
    domains = len(train_data)
    indices = list(np.random.permutation(domains))

    if leave_one_out:
        indices = indices[:-1], indices[-1:]  # random_split to meta-train and meta-test
    else:
        indices = [[indices[i]] for i in range(domains)]

    train_inputs_lists = []
    for idx in indices:  #
        inputs = get_image_and_label(train_data, idx, device)
        train_inputs_lists.append(inputs)
    return train_inputs_lists


def init_network(meta_model, meta_lr, previous_opt=None, momentum=0.9, Adam=False, beta1=0.9, beta2=0.999, device=None):
    fast_model = copy.deepcopy(meta_model).train()
    if device is not None:
        fast_model.to(device)
    if Adam:
        fast_opts = torch.optim.Adam(fast_model.parameters(), lr=meta_lr, betas=(beta1, beta2), weight_decay=5e-4)
    else:
        fast_opts = torch.optim.SGD(fast_model.parameters(), lr=meta_lr, weight_decay=5e-4, momentum=momentum)

    if previous_opt is not None:
        fast_opts.load_state_dict(previous_opt.state_dict())
    return fast_model, fast_opts


def load_state(new_opts, old_opts):
    [old.load_state_dict(new.state_dict()) for old, new in zip(old_opts, new_opts)]


def update_meta_model(meta_model, fast_param_list, optimizers, meta_lr=1):
    meta_params, meta_states = get_parameters(meta_model)

    optimizers.zero_grad()

    # update grad
    for k in meta_params.keys():
        new_v, old_v = 0, meta_params[k]
        for m in fast_param_list:
            new_v += m[0][k]
        new_v = new_v / len(fast_param_list)
        meta_params[k].grad = ((old_v - new_v) / meta_lr).data

    # update with original optimizers
    optimizers.step()

    # for k in meta_states.keys():
    #     new_v, old_v = 0, meta_states[k]
    #     for m in fast_param_list:
    #         new_v += m[1][k]
    #         break
    #     #new_v = new_v / len(fast_param_list)
    #     meta_states[k].data = new_v.data


def avg_meta_model(meta_model, fast_param_list):
    meta_params, meta_states = get_parameters(meta_model)

    # update grad
    for k in meta_params.keys():
        new_v, old_v = 0, meta_params[k]
        for m in fast_param_list:
            new_v += m[k]
        new_v = new_v / len(fast_param_list)
        meta_params[k].data = new_v.data


def add_grad(meta_model, fast_model, factor):
    meta_params, meta_states = get_parameters(meta_model)
    fast_params, fast_states = get_parameters(fast_model)
    grads = []
    for k in meta_params.keys():
        new_v, old_v = fast_params[k], meta_params[k]
        if meta_params[k].grad is None:
            meta_params[k].grad = ((old_v - new_v) * factor).data  # if data is not used, the tensor will cause the memory leak
        else:
            meta_params[k].grad += ((old_v - new_v) * factor).data  # if data is not used, the tensor will cause the memory leak
        grads.append((old_v - new_v))
    return grads


def compare_two_dicts(d1, d2):
    flag = True
    for k in d1.keys():
        if not ((d1[k] - d2[k]).abs().max() < 1e-7):
            print(k, (d1[k] - d2[k]).abs().max())
            flag = False
    return flag


def correlation(grad1, grad2, cos=False):
    all_sim = []
    for g1, g2 in zip(grad1, grad2):
        if cos:
            sim = F.cosine_similarity(g1.view(-1), g2.view(-1), 0)
        else:
            sim = (g1 * g2).sum()
        all_sim.append(sim)
    all_sim = torch.stack(all_sim)
    return all_sim.mean()


def regularize_params(meta_model, params, opts, weight):
    def get_direction(param1, param2):
        dirs = []
        for p1, p2 in zip(param1, param2):
            dirs.append(p2 - p1)
        return dirs

    def get_mean(dirs):
        mean_dir = []
        for ls in zip(*dirs):
            v = 0
            for m in ls:
                v += m
            v = v / len(ls)
            mean_dir.append(v)
        return mean_dir

    meta_param = get_parameters(meta_model)[0]

    # get gradient direction from each models
    dirs = [get_direction(meta_param.values(), param.values()) for param in params]

    # get mean gradient direction
    mean_dir = get_mean(dirs)

    # measure distance between mean and other directions
    dists = []
    for i in range(len(dirs)):
        for j in range(len(dirs)):
            if j > i:
                dists.append(correlation(dirs[i], dirs[j], cos=True))
    dists = 1 - torch.stack(dists).mean()
    zero_and_update(opts, dists * weight)  # w/o, w/
    return dists


def mixup_parameters(params, num=2, alpha=1):
    assert num <= len(params)
    selected_list = np.random.permutation(len(params))[:num]
    if alpha > 0:
        ws = np.float32(np.random.dirichlet([alpha] * num))  # Random mixup params
    else:
        ws = [1 / num] * num  # simply average model
    params = [params[i] for i in selected_list]
    new_param = {}
    for name in params[0].keys():
        new_p = 0
        for w, p in zip(ws, params):
            new_p += w * p[name]
        new_param[name] = new_p
    return new_param, selected_list


def average_models(models):
    params = [get_parameters(m)[0] for m in models]
    new_param, _ = mixup_parameters(params, num=len(params), alpha=0)
    new_model = copy.deepcopy(models[0])
    averaged_model = put_parameters(new_model, new_param, None)
    return averaged_model


def get_consistency_loss(logits_clean, logits_aug=None, T=4, weight=2):
    if logits_aug is None:
        length = len(logits_clean)
        logits_clean, logits_aug = logits_clean[length // 2:], logits_clean[:length // 2]
    logits_clean = logits_clean.detach()
    p_clean, p_aug = (logits_clean / T).softmax(1), (logits_aug / T).softmax(1)
    p_mixture = ((p_aug + p_clean) / 2).clamp(min=1e-7, max=1).log()
    loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') + F.kl_div(p_mixture, p_aug, reduction='batchmean')) * weight
    return loss



@contextmanager
def meta_learning_MAML(meta_model):
    fast_model = copy.deepcopy(meta_model)
    params, states = get_parameters(meta_model)
    fast_model = put_parameters(fast_model, params, states).train()

    def meta_step(self, meta_loss, meta_lr, use_second_order=False, ignore_keys=None):
        params = get_parameters(self)[0]
        params = update_parameters(meta_loss, params, meta_lr, use_second_order=use_second_order, ignore_keys=ignore_keys)
        put_parameters(self, params, None)

    fast_model.meta_step = types.MethodType(meta_step, fast_model)  # assign method to the instance
    yield fast_model
    del fast_model, params, states


def freeze(model, name, freeze, reverse=False):
    for n, p in model.named_parameters():
        if not reverse:
            if name in n:
                p.requires_grad = freeze
        else:
            if name not in n:
                p.requires_grad = freeze


class ParamOperator():
    def extract_params_with_grad(self, model):
        params = get_parameters(model)[0]
        param_list = [p.view(-1) for p in params.values() if p.grad is not None]
        grad_list = [p.grad.view(-1) for p in params.values() if p.grad is not None]
        return torch.cat(param_list), torch.cat(grad_list)

    def put_parameters(self, model, params, data=False):
        new_param = self.unravel_params(model, params)
        put_parameters(model, new_param, None, data)

    def extract_params(self, model, split=False):
        params = get_parameters(model)[0]
        param_list = [p.view(-1) for p in params.values()]
        if split:
            return param_list
        else:
            return torch.cat(param_list)

    def proj(self, v1, v2):
        return v1 - torch.dot(v1, v2) / torch.dot(v2, v2) * v2

    def unravel_params(self, model, params, with_grad=False):
        old_params = get_parameters(model)[0]
        new_param, offset = {}, 0
        for name, param in old_params.items():
            if with_grad and param.grad is None:
                continue
            shape, num = param.shape, param.shape.numel()
            new_param[name] = params[offset:offset + num].view(*shape).clone()
            offset += num
        return new_param

    def put_grads(self, model, grad_vec, multiplier=1.0):
        grad_vec = grad_vec.clone()
        idx = 0
        for name, param in model.named_parameters():
            arr_shape = param.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            if grad_vec is not None:
                param.grad = grad_vec[idx:idx + size].reshape(arr_shape)
            else:
                param.grad = param.grad * multiplier
            idx += size
