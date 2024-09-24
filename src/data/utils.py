import numpy as np
from typing import Dict
import torch

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .slimpajama import get_slimpajama_data
from .cosmopedia import get_cosmopedia_data

def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "arxiv2000":
        return get_arxiv_2000()
    if args.dataset == "arxiv":
        return get_arxiv_full()
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        return {'train': train_data, 'val': val_data}
    if args.dataset == 'openwebtext2':
        return get_openwebtext2_data()
    if args.dataset == "slimpajama":
        return get_slimpajama_data()
    if args.dataset == 'cosmopedia':
        return get_cosmopedia_data()
    else:
        raise NotImplementedError(f"Unknown dataset key '{args.dataset}'")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length
        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64)
        )
        return x, y


class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, sequence_length, padding_token=50256):
        # 50256 is the EOT token for gpt2 tokenizer. we also use it as the tokenized padding token here
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length
        # self.prefix_token = prefix_token
        self.padding_token = padding_token

        self.len_questions, self.len_answers, self.starting_indices = [], [], []
        self._get_question_lengths()

    def _get_question_lengths(self):
        num_questions = self.data[0]
        current_ind = num_questions*2 + 1
        for i in range(num_questions):
            self.starting_indices.append(current_ind)
            self.len_questions.append(self.data[i*2 + 1])
            self.len_answers.append(self.data[i*2 + 2])
            current_ind += self.len_questions[-1] + self.len_answers[-1]

    def __len__(self):
        return self.data[0]

    def __getitem__(self, idx):

        starting = self.starting_indices[idx]
        len_sample = self.len_questions[idx] + self.len_answers[idx]
        last_idx = min(self.sequence_length, len_sample) + starting

        x = torch.from_numpy(self.data[starting:last_idx].astype(np.int64))
        y = torch.from_numpy(self.data[starting+1:last_idx + 1].astype(np.int64))

        #  TODO: here, the prediction for last token doesn't make any sense. Check what to do with it!

        if len_sample < self.sequence_length:
            x = torch.cat((x, torch.full((self.sequence_length - len(x),), self.padding_token, dtype=torch.int64)))
            y = torch.cat((y, torch.full((self.sequence_length - len(y),), self.padding_token, dtype=torch.int64)))

        return (x,
                y,
                min(self.sequence_length-1, self.len_questions[idx].astype(np.int64)),
                min(len_sample.astype(np.int64), self.sequence_length))


def get_dataloader(data, sequence_length, batch_size, dataset='slimpajama', seed=0, distributed_backend=None):
    """Create a DataLoader for the given data. If distributed_backend is provided and is truly
    distributed (world size > 1), the DataLoader will be created with a DistributedSampler that
    splits the data across the processes (in conjunction with DDP).
    Otherwise, use a RandomSampler with the specified seed.

    Returns both the dataloader and the sampler.
    """
    if dataset == 'cosmopedia':
        dataset = QADataset(data, sequence_length=sequence_length)
    else:
        dataset = Dataset(data, sequence_length=sequence_length)

    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
    )

    # breakpoint()
    return loader, sampler
