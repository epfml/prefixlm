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

    loss_dict_val, acc_dict, perplexity_dict = {}, {}, {}
    loss_dict_val["all"] = []
    acc_dict["all"] = []
    sizes = [0 for _ in range(extra_args.max_context_ratio)]

    num_samples = 0
    mx_context = extra_args.max_context_ratio
    last_loss_token = extra_args.sequence_length
    original_seq_len = extra_args.sequence_length

    if extra_args.long_context:
        last_loss_token *= mx_context
        for i in range(mx_context):
            loss_dict_val[f"long_context_{i}"] = []
            acc_dict[f"long_context_{i}"] = []

    for _ in range(max_num_batches):
        causal_pos = 0

        if extra_args.dataset == 'cosmopedia':
            x, y, causal_pos, last_loss_token = get_batch(data_val_iter, extra_args.dataset, device=device)
        else:
            x, y = get_batch(data_val_iter, extra_args.dataset, device=device)
            if extra_args.prefixlm_eval:
                causal_pos = uniform_step(100, 0.9, original_seq_len)

        if extra_args.prefix_token:
            assert causal_pos is not None
            x, y = add_special_token(x, y, causal_pos)

        eval_normalizer = None
        if extra_args.eval_normalizer:
            eval_normalizer = uniform_step(100, 0.9, original_seq_len)

        with ctx:
            outputs = model(x,
                            targets=y,
                            prefixlm=extra_args.prefixlm_eval,
                            last_loss_token=last_loss_token,
                            get_logits=True,
                            causal_pos=causal_pos,
                            eval_normalizer=eval_normalizer)

        logit_mask = outputs['logit_mask']
        num_samples += outputs['num_samples']
        # if outputs['num_samples'] != 65536:
        #     print('num_samples is not 65536')
        #     breakpoint()
        # breakpoint()

        val_loss = outputs['loss']
        loss_dict_val["all"].append(val_loss.sum())
        right_preds = (outputs['logits'].argmax(-1) == y)[logit_mask].float()
        acc_dict["all"].append(torch.sum(right_preds))

        # breakpoint()
        if extra_args.long_context:
            batch_size = x.shape[0]
            for i in range(mx_context):
                # breakpoint()
                start = max(0, i*original_seq_len - causal_pos)
                #  making sure that we don't get the loss of the prefix
                end = max((i+1)*original_seq_len - causal_pos, 0)
                loss_dict_val[f"long_context_{i}"].append(val_loss.view(batch_size, -1)[:, start:end].sum())
                acc_dict[f"long_context_{i}"].append(right_preds.view(batch_size, -1)[:, start:end].sum())
                sizes[i] += outputs['logit_mask'][:,start:end].sum()


    acc_dict["all"] = torch.stack(acc_dict["all"]).sum()/num_samples
    loss_dict_val["all"] = torch.stack(loss_dict_val["all"]).sum()/num_samples

    perplexity_dict["all"] = 2.71828 ** loss_dict_val["all"]
    if extra_args.long_context:
        for i in range(mx_context):
            acc_dict[f"long_context_{i}"] = torch.stack(acc_dict[f"long_context_{i}"]).sum()/sizes[i]
            loss_dict_val[f"long_context_{i}"] = torch.stack(loss_dict_val[f"long_context_{i}"]).sum()/sizes[i]
            perplexity_dict[f"long_context_{i}"] = 2.71828 ** loss_dict_val[f"long_context_{i}"]

    if abs((acc_dict["long_context_0"] + acc_dict["long_context_1"] + acc_dict["long_context_2"] + acc_dict["long_context_3"])/4 - acc_dict["all"]) > 1e-4:
        print('accuracies dont match')
        breakpoint()
    if abs((loss_dict_val["long_context_0"] + loss_dict_val["long_context_1"] + loss_dict_val["long_context_2"] + loss_dict_val["long_context_3"])/4 -  loss_dict_val["all"]) > 1e-4:
        print('losses dont match')
        breakpoint()

    return acc_dict, loss_dict_val, perplexity_dict


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
