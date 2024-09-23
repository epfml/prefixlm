import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, dataset, device="cpu"):
    if dataset == 'cosmopedia':
        x, y, loc = next(dataloader)
    else:
        x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        if dataset == 'cosmopedia':
            loc = loc.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        if dataset == 'cosmopedia':
            loc = loc.to(device)

    if dataset == 'cosmopedia':
        return x, y, loc

    return x, y


def uniform_step(step_loc=100, step_prob=0.9, seq_length=512):
    if torch.rand(1).item() < step_prob:
        return torch.randint(1, step_loc, (1,)).item()
    else:
        return torch.randint(step_loc, seq_length, (1,)).item()


def add_special_token(x, y, causal_pos):
    modified_x = torch.zeros_like(x)
    modified_x[:, :causal_pos] = x[:, :causal_pos]
    modified_x[:, causal_pos + 1:] = x[:, causal_pos:x.shape[1] - 1]
    modified_x[:, causal_pos] = 50257
    modified_y = torch.zeros_like(y)
    modified_y[:, :causal_pos] = y[:, :causal_pos]
    modified_y[:, causal_pos] = y[:, causal_pos-1]
    modified_y[:, causal_pos + 1:] = y[:, causal_pos:y.shape[1]-1]
    return modified_x, modified_y


@torch.no_grad()
def eval(model, data_val_iter, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []
    num_samples = 0
    for _ in range(max_num_batches):
        if extra_args.dataset == 'cosmopedia':
            x, y, loc = get_batch(data_val_iter, extra_args.dataset, device=device)
        else:
            x, y = get_batch(data_val_iter, extra_args.dataset, device=device)

        causal_pos=0
        if extra_args.dataset == 'cosmopedia':
            causal_pos = loc
            if extra_args.prefix_token:
                x, y = add_special_token(x, y, causal_pos)
        elif extra_args.prefixlm_eval:
            # causal_pos = torch.randint(0, t, (1,)).item()
            causal_pos = uniform_step(100, 0.9, x.shape[1])
            if extra_args.prefix_token:
                x, y = add_special_token(x, y, causal_pos)

        with ctx:
            eval_normalizer = 0
            if not extra_args.prefixlm_eval and extra_args.dataset != 'cosmopedia':
                eval_normalizer = uniform_step(100, 0.9, x.shape[1])
            # breakpoint()
            outputs = model(x, targets=y, get_logits=True, causal_pos=causal_pos, eval_normalizer=eval_normalizer)

        logit_mask = outputs['logit_mask']
        num_samples += logit_mask.int().sum().item()

        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append(torch.sum((outputs['logits'].argmax(-1) == y).float()[logit_mask]))

    val_acc = torch.stack(acc_list).sum().item()/num_samples
    val_loss = torch.stack(loss_list_val).sum().item()/num_samples
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
