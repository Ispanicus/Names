import tempfile
from pathlib import Path
from multiprocessing.pool import Pool

import pandas as pd
from transformers import PreTrainedTokenizerFast

from names.util import get_data_path, just_give_me_all_the_shit
import shutil
import json

NAMES_W_4_CHAR_PLUS = just_give_me_all_the_shit('firstName').name[lambda x: x.agg(len) > 3].unique()

INPUT = get_data_path('data/raw_pretrained')
OUTPUT = get_data_path('data/pretrained')

def clone_files():
    OUTPUT.mkdir(exist_ok=True)
    for f in INPUT.glob('*'):
        new_dir = OUTPUT / f.name
        if new_dir.exists():
            shutil.rmtree(new_dir)
        shutil.copytree(f, new_dir)

def clean_file(f):
    print(f, flush=True)
    added_tokens, merges = load_vocab_and_merges(f)

    df = pd.DataFrame(zip(added_tokens, merges), columns=['vocab', 'merges'])
    split_merges = df.merges.str.split().explode()

    banned_names = df[df.vocab.isin(NAMES_W_4_CHAR_PLUS)]
    banned_idx = banned_names.index.to_list()
    banned_set = set()
    for idx in banned_idx:
        banned = df.loc[idx, 'vocab']
        derivations = split_merges[split_merges == banned].index
        for d_idx in derivations:
            if d_idx not in banned_set:
                banned_set.add(d_idx)
                banned_idx.append(d_idx)
    print(f, banned_names.vocab, df.loc[list(banned_set)], sep='\n')

    clean_df = df.loc[df.index.difference(banned_idx)]

    update_vocab_and_merges(f, clean_df.vocab.to_list(), new_merges = clean_df.merges.to_list())

def update_vocab_and_merges(f, new_vocab: list, new_merges: list, vocab_size: int = None):
    data = json.load(open(f))

    vocab = list(data['model']['vocab'])
    start_idx = next((i for i in range(33) if len(vocab[i]) == 2))
    base_vocab = vocab[:start_idx]

    new_vocab_size = None if vocab_size is None else vocab_size - start_idx
    new_vocab = {v: idx for idx, v in enumerate(base_vocab + new_vocab[:new_vocab_size])} # 32ish base tokens

    data['model']['vocab'] = new_vocab
    data['model']['merges'] = new_merges[:new_vocab_size]

    json.dump(data, open(f, 'w'), indent=4)

def clean_files():
    todo = list(OUTPUT.glob('*/tokenizer.json'))
    with Pool(processes=6) as pool: # 3 families, longest takes around 
        list(pool.imap_unordered(clean_file, todo))
    

def load_vocab_and_merges(f: Path):
    assert f.name == 'tokenizer.json'
    data = json.load(open(f))

    vocab = list(data['model']['vocab'])
    merges = list(data['model']['merges'])

    # base_tokens = vocab[:32] # some might not start at 32 if they have fewer unique chars
    start_idx = next((i for i in range(33) if len(vocab[i]) == 2))
    added_tokens = vocab[start_idx:]
    assert len(added_tokens) == len(merges), f'{len(added_tokens), len(merges) = }'

    return added_tokens, merges

def load_model_w_first_n_tokens(f: Path, n: int | None):
    assert n is None or n > 32, 'cannot reduce below base tokens'
    added_tokens, merges = load_vocab_and_merges(f / 'tokenizer.json')

    with tempfile.TemporaryDirectory() as fp:
        fp += '1'
        Path(fp).mkdir()
        for path in f.glob('*'):
            shutil.copy(path, Path(fp) / path.name)

        tok_file = Path(fp) / 'tokenizer.json'
        update_vocab_and_merges(tok_file, added_tokens, merges, vocab_size=n)
        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(fp)
    return fast_tokenizer

if __name__ == '__main__':
    clone_files()
    clean_files()
