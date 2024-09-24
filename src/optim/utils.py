import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, dataset, device="cpu"):
    if dataset == 'cosmopedia':
        x, y, loc, last_loss_token = next(dataloader)
    else:
        x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        if dataset == 'cosmopedia':
            loc = loc.pin_memory().to(device, non_blocking=True)
            last_loss_token = last_loss_token.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        if dataset == 'cosmopedia':
            loc = loc.to(device)
            last_loss_token = last_loss_token.to(device)

    if dataset == 'cosmopedia':
        return x, y, loc, last_loss_token

    return x, y


def uniform_step(step_loc=100, step_prob=0.9, seq_length=512):
    if torch.rand(1).item() < step_prob:
        return torch.randint(1, step_loc, (1,)).item()
    else:
        return torch.randint(step_loc, seq_length, (1,)).item()


def add_special_token(x, y, causal_pos):
    # first shift every token to the right
    modified_x = torch.zeros_like(x)
    modified_x[:, 1:] = x[:, :x.shape[1]-1]

    # then return the tokens in the prefix to their original index
    column_indices = torch.arange(x.shape[1], device=x.device)
    mask = column_indices.unsqueeze(0) < causal_pos.unsqueeze(1)
    modified_x[mask] = x[mask]
    # finally, add the special token to the causal position
    row_indices = torch.arange(x.shape[0], device=x.device)
    tensor_to_assign = torch.full_like(modified_x[row_indices, causal_pos], 50257)
    modified_x[row_indices, causal_pos] = tensor_to_assign

    modified_y = torch.zeros_like(y)
    modified_y[:, 1:] = y[:, :y.shape[1] - 1]
    modified_y[mask] = y[mask]

    return modified_x, modified_y


@torch.no_grad()
def eval(model, data_val_iter, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []
    num_samples = 0
    for _ in range(max_num_batches):
        causal_pos = None
        last_loss_token = extra_args.sequence_length
        if extra_args.dataset == 'cosmopedia':
            x, y, causal_pos, last_loss_token = get_batch(data_val_iter, extra_args.dataset, device=device)
        else:
            x, y = get_batch(data_val_iter, extra_args.dataset, device=device)
            if extra_args.prefixlm_eval:
                causal_pos = uniform_step(100, 0.9, x.shape[1])

        if extra_args.prefix_token:
            assert causal_pos is not None
            x, y = add_special_token(x, y, causal_pos)

        with ctx:
            eval_normalizer = None
            if not extra_args.prefixlm_eval and extra_args.dataset != 'cosmopedia':
                eval_normalizer = uniform_step(100, 0.9, x.shape[1])

            outputs = model(x,
                            targets=y,
                            last_loss_token=last_loss_token,
                            get_logits=True,
                            causal_pos=causal_pos,
                            eval_normalizer=eval_normalizer)

        logit_mask = outputs['logit_mask']
        num_samples += outputs['num_samples']

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
