from contextlib import nullcontext
from data.utils import get_dataloader

import torch
import torch.nn.functional as F
import wandb
import time
import itertools
import copy
import random
import os
import numpy as np
from .utils import eval, get_batch, save_checkpoint, uniform_step, add_special_token


def train_base(model, opt, data, data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq,
               ckpt_path, distributed_backend, extra_args, itr=0, rng_state_dict=None):

    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)
    best_val_loss, text_table = float(
        'inf'), None  # best_val_loss not used atm, early stopping not recommended but possible
    substep = itr * acc_steps
    data["train"], train_sampler = get_dataloader(
        data["train"],
        sequence_length=sequence_length,
        batch_size=batch_size,
        dataset=extra_args.dataset,
        seed=data_seed,
        distributed_backend=distributed_backend,
    )
    sampler_state_before_iter = None

    data["val"], val_sampler = get_dataloader(
        data["val"],
        sequence_length=sequence_length*extra_args.max_context_ratio if extra_args.long_context else sequence_length,
        batch_size=int(batch_size/extra_args.max_context_ratio),
        dataset=extra_args.dataset,
        seed=data_seed,
    )

    num_substeps_per_epoch = len(data["train"])
    train_epochs = substep // num_substeps_per_epoch

    print('substeps per epoch:', num_substeps_per_epoch)

    if rng_state_dict is not None and rng_state_dict.get("train_sampler_state", None) is not None:
        train_sampler.generator.set_state(rng_state_dict["train_sampler_state"])
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(train_epochs)
    else:
        sampler_state_before_iter = train_sampler.generator.get_state()
    data_train_iter = iter(data["train"])

    # for val data we don't care about epochs? just cycle through (no need to set_epoch to reshuffle)
    data_val_iter = itertools.cycle(data["val"])

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}

    if extra_args.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)  # requires pytorch 2.0+

    model.train()
    t0 = time.time()

    if rng_state_dict is not None:
        torch.set_rng_state(rng_state_dict["cpu_rng_state"])
        torch.cuda.set_rng_state(rng_state_dict["gpu_rng_state"])
        np.random.set_state(rng_state_dict["numpy_rng_state"])
        random.setstate(rng_state_dict["py_rng_state"])
    for _ in range(substep % num_substeps_per_epoch):
        get_batch(data_train_iter, extra_args.dataset, device=extra_args.device)

    seen_samples = 0
    generation_string = extra_args.eval_seq_prefix

    while itr < iterations:
        # print('iter:', itr)
        for microstep_idx in range(acc_steps):  # gradient accumulation
            causal_pos = 0
            num_samples = 0
            last_loss_token = sequence_length
            if extra_args.dataset == 'cosmopedia':
                x, y, causal_pos, last_loss_token = get_batch(data_train_iter, extra_args.dataset, device=extra_args.device)
                if itr == 0:
                    # breakpoint()
                    tokenizer = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer
                    generation_string = tokenizer.decode(x[0, :causal_pos[0]].to('cpu').numpy())
            else:
                x, y = get_batch(data_train_iter, extra_args.dataset, device=extra_args.device)
                if extra_args.prefixlm_train:
                    causal_pos = uniform_step(100, 0.9, x.shape[1])

            if extra_args.prefix_token:
                assert causal_pos is not None
                x, y = add_special_token(x, y, causal_pos)

            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx,
                                                                           gradient_accumulation_steps=acc_steps):
                    outputs = model(x,
                                    targets=y,
                                    pe=extra_args.pe,
                                    prefixlm=extra_args.prefixlm_train,
                                    last_loss_token=last_loss_token,
                                    causal_pos=causal_pos,
                                    window=extra_args.train_window,
                                    itr=itr)
                    num_samples += outputs['num_samples']

                    #  TODO: make this right for the case that we put causal_pos = 0

            seen_samples += num_samples
            # This wouldn't lead to an accurate loss calculation, but it holds in expectation and is a good estimation
            loss = outputs['loss'].sum() / (num_samples * acc_steps)
            loss.backward()
            substep += 1
            if substep % len(data["train"]) == 0:
                train_epochs += 1
                print(f"Train epoch {train_epochs} done (full pass over training data)")
                if hasattr(train_sampler, "set_epoch"):
                    # set epoch for reshuffling between epochs
                    train_sampler.set_epoch(train_epochs)
                    sampler_state_before_iter = None
                else:
                    sampler_state_before_iter = train_sampler.generator.get_state()
                data_train_iter = iter(data["train"])

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        # Move optimizer state to the correct device
        # for param_group in opt.param_groups:
        #     for p in param_group['params']:
        #         p.data = p.data.to(torch.device(f'cuda:{torch.distributed.get_rank()}'))
        #         if p.grad is not None:
        #             print('there is grad that is not None')
        #             p.grad.data = p.grad.data.to(torch.device(f'cuda:{torch.distributed.get_rank()}'))

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1
        # breakpoint()
        if itr % eval_freq == 0 or itr == iterations:  # from here it's only evaluation code, all the training is above

            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep // num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item() * acc_steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr

                eval_steps = (
                    24*extra_args.max_context_ratio if itr < iterations else len(data["val"])
                )
                val_acc, val_loss, val_perplexity = eval(
                    model,
                    data_val_iter,
                    extra_args,
                    device=extra_args.device,
                    max_num_batches=eval_steps,
                    ctx=type_ctx,
                    itr=itr
                )

                print_string = (f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss['all'].item():.3f},"
                                f" pp={val_perplexity['all']:.2f}, acc={val_acc['all']:3f}")
                print_string += f" [time per itr] {dt * 1000 / eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)


                if extra_args.wandb:
                    logs = {
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss["all"].item(),
                        "val/perplexity": val_perplexity["all"].item(),
                        "val/acc": val_acc["all"].item(),
                        "lr": current_lr,
                        "seen-samples": seen_samples
                    }

                    for key in val_loss:
                        if key != "all":
                            logs[f"val_loss/{key}"] = val_loss[key].item()
                            logs[f"val_perplexity/{key}"] = val_perplexity[key].item()
                            logs[f"val_acc/{key}"] = val_acc[key].item()

                    if itr == iterations:
                        logs["val/final-ppl"] = val_perplexity
                        logs["val/final-acc"] = val_acc
                        logs["val/final-loss"] = val_loss

                    wandb.log(logs)

                    # if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                    #     if text_table is None:
                    #         text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                        # out_str = distributed_backend.get_raw_model(model).generate_from_string(
                        #     generation_string, max_new_tokens=100, temperature=0.9, top_k=None)
                        # print('out_str is: ', out_str)
                        # text_table.add_data(itr, val_perplexity, out_str)
                        # # why a copy? see github.com/wandb/wandb/issues/2981
                        # wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

                model.train()
                t0 = time.time()
        if distributed_backend.is_master_process():
            if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
                print(f"saving checkpoint to {os.path.dirname(ckpt_path)}/ckpt_{itr}.pt")
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                cpu_rng_state=torch.get_rng_state(),
                                gpu_rng_state=torch.cuda.get_rng_state(),
                                numpy_rng_state=np.random.get_state(),
                                py_rng_state=random.getstate(),
                                train_sampler_state=sampler_state_before_iter,  # I initialized it with None, does this work?
                                ckpt_path=os.path.join(os.path.dirname(ckpt_path), f"ckpt_{itr}.pt"))

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats
