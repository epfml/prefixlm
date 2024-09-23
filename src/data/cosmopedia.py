import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

OWT2_DATA_PATH = os.path.join(os.path.dirname(__file__), "datasets/cosmopedia/")
tknzr = tiktoken.get_encoding("gpt2")


def get_cosmopedia_data(num_proc=40):

    if not os.path.exists(os.path.join(OWT2_DATA_PATH, 'train.bin')):
        os.makedirs(OWT2_DATA_PATH, exist_ok=True)
        dataset = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train")
        split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')

        def process(example):
            ids_text = tknzr.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
            ids_prompt = tknzr.encode_ordinary(example['prompt'])
            ids_text.append(tknzr.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
            ids_prompt.append(tknzr.eot_token)

            ids = ids_prompt + ids_text
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids), 'len_q': len(ids_prompt), 'len_a': len(ids_text)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=['text', 'prompt'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            # add number of questions, then pairs of (length of Q, length of A), then dataset
            print('in the beginning of for loop the len is', len(dset))
            arr_len = np.sum(dset['len'], dtype=np.int64) + 1 + len(dset)*2
            print('arr_len is', arr_len)
            filename = os.path.join(OWT2_DATA_PATH, f'{split}.bin')
            dtype = np.uint32  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            print('the size of array is ', len(arr))
            total_batches = 1024


            #  writing the number of Q and As
            arr[0] = len(dset)
            print('the number of samples is', len(dset), arr[0])

            dset_np = dset.with_format('numpy')

            arr[1:2*len(dset)+1:2] = dset_np['len_q']
            arr[2:2*len(dset)+2:2] = dset_np['len_a']

            idx = 1 + len(dset)*2
            print('the starting index for writing the data is', idx)
            print('the fist number in the array before for loop', arr[0])

            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)


            print('finish idx is', idx)
            print('the first number in the array before flushing is:', arr[0])
            arr.flush()

    train_data = np.memmap(os.path.join(OWT2_DATA_PATH, 'train.bin'), dtype=np.uint32, mode='r')
    val_data = np.memmap(os.path.join(OWT2_DATA_PATH, 'val.bin'), dtype=np.uint32, mode='r')

    print('len train data before return', train_data[0])

    return {'train': train_data, 'val': val_data}

