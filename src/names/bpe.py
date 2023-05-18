import collections
import shutil
import math
import string
from typing import NamedTuple

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

from names.huffman_encoder import huffman_encoder
from names.util import get_data_path, timeit
from names.vocab_cleaner import load_model_w_first_n_tokens, clean_file

SEP = ' ' # Explicitly state seperater
MAX_TOKENS = 32768

class Result(NamedTuple):
    expected_bits_per_name: float
    expected_nan: float

class Output(NamedTuple):
    group_name: str | tuple[str, str]
    naive_letter: Result
    huff_letter: Result
    huff_name: Result
    bpe: Result
    bpe_vocab: int

def deserialize_fast_tokenizer(name: str, n_tokens: int | None) -> PreTrainedTokenizerFast:
    name = name.strip('1234567890')
    fast_tokenizer = load_model_w_first_n_tokens(get_data_path(f'data/pretrained/{name}'), n_tokens)
    if n_tokens is not None and fast_tokenizer.vocab_size != n_tokens:
        raise RuntimeError(f'Too few valid tokens possible! {name = } {fast_tokenizer.vocab_size = }, {n_tokens = }')
    return fast_tokenizer

def _deserialize_raw_fast_tokenizer(name: str, n_tokens: int) -> PreTrainedTokenizerFast:
    raise RuntimeError('not useful?')
    name = name.strip('1234567890')
    return load_model_w_first_n_tokens(get_data_path(f'data/raw_pretrained/{name}'), n_tokens)

def _serialize_raw_fast_tokenizer(fast_tokenizer: PreTrainedTokenizerFast, name: str):
    name = name.strip('1234567890')
    assert fast_tokenizer.vocab_size > 1000, fast_tokenizer.vocab_size # must at least be able to create 1000 byte pairs for 1000 names
    path = get_data_path('data/raw_pretrained') / name
    fast_tokenizer.save_pretrained(path)

def train_BPE_wrapper(name_seq_vocab):
    ''' for multi processing '''
    name, seq, vocab = name_seq_vocab #name: category Romance, vocab: vocab size, seq: sequence of names to train on, Series?
    try:
        return (name, deserialize_fast_tokenizer(name, vocab))
    except:
        print('BPE training', name, flush=True)
        try:
            fast_tokenizer = _deserialize_raw_fast_tokenizer(name, MAX_TOKENS)
        except:
            fast_tokenizer = train_BPE(seq, MAX_TOKENS)
            _serialize_raw_fast_tokenizer(fast_tokenizer, name)

        # clean
        
        path = get_data_path('data/raw_pretrained') / name
        cleaned_path = get_data_path('data/pretrained') / name
        if cleaned_path.exists():
            shutil.rmtree(cleaned_path)
        shutil.copytree(path, cleaned_path)
        clean_file(cleaned_path / 'tokenizer.json')

        return (name, deserialize_fast_tokenizer(name, vocab))


def brute_force_best_BPE(name, train_df, test_df) -> dict[int, float]:
    '''
    This might be improved by pruning
    Big picture: Decrement e.g. "chri" scores by the frequency of "chris". Then pick N largest
    Details:
        After getting a BPE vocab dict[word, count], sort by len(word) long->short
            For each (long) word, find all subwords. 
            These must include the counts of the longer word. 
            Decrement with count of the longer word
        Re-filter the vocab dict, picking with highest "freq - subword_freq"
        Must ensure that we keep all 1-letter words
    '''
    train_name_seq = df_to_name_seq(train_df)

    candidates = []
    for n_bits in range(6, 15):
        try:
            candidates.append(
                evaluate_BPE(
                    test_seq = test_df.name,
                    weight_seq = test_df.normed_count,
                    fast_tokenizer=train_BPE_wrapper((name, train_name_seq, 2**n_bits))[1]
                )
            )
        except RuntimeError:
            break
    result, vocab_size = min(candidates)
    return result, vocab_size

def df_to_name_seq(train_df) -> pd.Series:
    return train_df.name.repeat(train_df.normed_count)

def df_to_name_str(test_df) -> str:
    return test_df.name.repeat(test_df.normed_count).str.cat(sep=SEP)

def train_BPE(train_name_seq: pd.Series, vocab_size: int) -> tuple[int, float]:
    use_bpe = True
    MODEL, TRAINER = (BPE, BpeTrainer) if use_bpe else (WordPiece, WordPieceTrainer)
    trainer = TRAINER(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)
    tokenizer = Tokenizer(MODEL(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(train_name_seq, trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    return fast_tokenizer

@timeit
def evaluate_BPE(test_seq: pd.Series, weight_seq: pd.Series, fast_tokenizer: PreTrainedTokenizerFast) -> Result:
    assert isinstance(test_seq, pd.Series)
    assert isinstance(weight_seq, pd.Series)
    assert isinstance(next(iter(test_seq)), str)
    n_names = weight_seq.sum()

    vocab_size = len(fast_tokenizer.get_vocab())
    bits_per_token = math.ceil(math.log2(vocab_size))

    encoding, _, _ = fast_tokenizer(test_seq.to_list()).values()
    subwords = pd.Series(encoding).repeat(weight_seq).reset_index(drop=True).explode()
    invalid = subwords[subwords == 0].index.unique()
    valid_subwords = subwords[~subwords.index.isin(invalid)]

    # We must multiply each subword by n_bits, since each subword is encoded using the same number of bits
    n_bits = len(valid_subwords) * bits_per_token
    return Result(n_bits / n_names, len(invalid) / n_names)

@timeit
def naive_alphabet_baseline(_, test_df: pd.DataFrame) -> Result:
    '''
    Assumes alphabet matches the regex pattern [a-z \-]
    Returns the expected number of bits required to encode randomly sampled names from df.name, with weight df.normed_count
    '''
    to_encode = df_to_name_seq(test_df)
    ALPHA_LEN = len(string.ascii_lowercase + ' -')
    CHAR_ENCODING_COST = math.ceil(math.log2(ALPHA_LEN))
    name_costs = CHAR_ENCODING_COST*(to_encode.agg(len) + len(SEP))
    return Result(name_costs.sum() / len(to_encode), 0)

@timeit
def huffman_letter_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Result:
    '''
    Uses huffman encoding to slightly improve baseline, by weighting common letters more
    Returns the expected number of bits required to encode randomly sampled names from df.name
    '''
    n_names = test_df.normed_count.sum()
    class Counter(collections.Counter):
        def __mul__(self, other):
            assert isinstance(other, int)
            return Counter({k: v*other for k, v in self.items()})

    letter_counts = ((train_df.name + ' ').apply(Counter) * train_df.normed_count).sum()

    test_df = test_df.reset_index(drop=True) # required for missing_names calculation
    to_encode = (df_to_name_seq(test_df) + ' ').agg(list).explode()

    n_bits, missing_names = huffman_encode_and_eval(letter_counts, to_encode)
    return Result(n_bits / n_names, missing_names / n_names)

@timeit
def huffman_name_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Result:
    '''
    Uses huffman encoding to slightly improve baseline, by weighting common letters more
    Returns the expected number of bits required to encode randomly sampled names from df.name
    '''
    to_encode = df_to_name_seq(test_df.reset_index(drop=True))
    name_counts = train_df.set_index('name').normed_count
    n_bits, missing_names = huffman_encode_and_eval(name_counts.to_dict(), to_encode)
    return Result(n_bits / len(to_encode), missing_names / len(to_encode))

def huffman_encode_and_eval(counts: dict, to_encode: pd.Series):
    mapping = huffman_encoder(counts)
    encoded = to_encode.map(mapping)

    null = encoded.isnull()
    missing_names = encoded.index[null].nunique() # Index contains name id
    encoded = encoded[~null]

    n_bits = encoded.agg(len).sum()
    return n_bits, missing_names

if __name__ == '__main__':
    fast_tokenizer = deserialize_raw_fast_tokenizer('full2048')
    invalid_name = 'X@'
    names = ['james', 'holly', invalid_name]
    encodings, _, _ = fast_tokenizer(names).values()
    print(f'{list(zip(names, encodings)) = }')
    print(f"{fast_tokenizer.tokenize(' '.join(names)) = }")