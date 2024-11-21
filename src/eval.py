import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb
import logging
import time

from tqdm import tqdm

import config as config
from data.utils import get_dataset, get_dataloader
import distributed as distributed
import models.utils as models_utils
from optim.utils import eval
from contextlib import nullcontext

def none_or_str(value):
    if value == 'None':
        return None
    return value


def get_args(args=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--checkpoint', type=none_or_str, required=True)
    parser.add_argument('--config_format', type=str, required=False)

    args, rem_args = parser.parse_known_args(args)

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            args.checkpoint, args.checkpoint_filename = os.path.split(args.checkpoint)
        else:
            args.checkpoint_filename = "ckpt.pt"

        with open(os.path.join(args.checkpoint, "summary.json")) as f:
            summary = json.load(f)

        for k, v in summary['args'].items():
            if k == "config_format" and args.config_format is not None:
                continue
            if k not in ["device", "dtype"]:
                setattr(args, k, v)

    return config.parse_args_with_format(format=args.config_format,
                                         base_parser=argparse.ArgumentParser(allow_abbrev=False), args=rem_args,
                                         namespace=args)

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True
    # breakpoint()

    args.distributed_backend = None

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.window = True
    # args.max_context_ratio = 2
    # args.sequence_length = 2048
    # args.long_context = False
    # args.window = False
    # args.batch_size = 32
    # args.acc_steps = 4
    args.device = torch.device(args.device)
    torch.cuda.set_device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")

    # if distributed_backend.is_master_process():
    #     prepare_dataset(args)
    distributed_backend.sync()

    data = get_dataset(args)  # data is a dict: {'train': train_tokenized, 'val': eval_tokenized}




    print(f"Num training tokens: {len(data['train'])}")
    print(f"Num validation tokens: {len(data['val'])}")

    data = np.array(data['val'])
    data, val_sampler = get_dataloader(
        data,
        sequence_length= args.sequence_length * args.max_context_ratio if args.long_context else args.sequence_length,
        batch_size=int(args.batch_size / args.max_context_ratio) if args.long_context else args.batch_size,
        dataset=args.dataset,
        seed=args.data_seed,
    )

    data_iter = iter(data)

    # model = models.make_model_from_args(args).to(args.device)
    model = models_utils.get_model(args).to(args.device)

    if args.checkpoint is not None:
        checkpoint = torch.load(os.path.join(args.checkpoint, args.checkpoint_filename))
        model.load_state_dict({x: y for x, y in checkpoint['model'].items() if "attn.bias" not in x and "wpe" not in x},
                              strict=False)

    model = distributed_backend.transform_model(model)

    print(f"\Evaluating model={args.model} \n{vars(args)}\n")

    # stats = evaluate(model, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length,
    #                  distributed_backend=distributed_backend,
    #                  extra_args=args)

    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)

    model.eval()
    # breakpoint()
    stats = eval(model, data_iter, args, device=args.device, max_num_batches=len(data), ctx=type_ctx)

    distributed_backend.finalize()

    first_half = stats[2]['first_half']
    long_context_0 = stats[2]['long_context_0']
    mean_long_context_0_4 = (stats[2]['long_context_0'].item() + stats[2]['long_context_1'].item() +
                             stats[2]['long_context_2'].item() + stats[2]['long_context_3'].item())/4
    mean_long_context_0_8 = (stats[2]['long_context_0'].item() + stats[2]['long_context_1'].item()+
                                    stats[2]['long_context_2'].item() + stats[2]['long_context_3'].item() +
                                    stats[2]['long_context_4'].item() + stats[2]['long_context_5'].item() +
                                    stats[2]['long_context_6'].item() + stats[2]['long_context_7'].item())/8

    print('first half', first_half, '\nlong context0', long_context_0, '\nlong context0-4', mean_long_context_0_4,
          '\nlong context0-8', mean_long_context_0_8)

    print(stats)

    return stats


if __name__ == "__main__":
    args = get_args()
    main(args)
