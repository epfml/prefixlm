import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, device="cpu"):
    x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

def uniform_step(step_loc=100, step_prob=0.9, seq_length=512):
    if torch.rand(1).item() < step_prob:
        return torch.randint(0, step_loc, (1,)).item()
    else:
        return torch.randint(step_loc, seq_length, (1,)).item()

@torch.no_grad()
def eval(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext(), prefixlm=False):
    assert model.training == False

    loss_list_val, acc_list = [], []
    num_samples = 0
    for _ in range(max_num_batches):
        x, y = get_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, prefixlm=prefixlm, eval_loss=True)
            causal_pos = outputs['causal_pos']

        num_samples += (y.shape[1] - causal_pos)*y.shape[0]

        val_loss = outputs['loss']
        # breakpoint()
        loss_list_val.append(val_loss)
        # breakpoint()
        acc_list.append((outputs['logits'].argmax(-1)[:,causal_pos:] == y[:,causal_pos:]).float().sum())

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
