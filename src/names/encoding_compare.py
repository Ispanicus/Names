from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from names.bpe import (Output, brute_force_best_BPE, df_to_name_seq,
                       evaluate_BPE, huffman_letter_baseline,
                       huffman_name_baseline, naive_alphabet_baseline,
                       train_BPE_wrapper, deserialize_fast_tokenizer)
from names.util import (FAMILY_ORDER, SUMMARY, get_data_path,
                        just_give_me_all_the_shit, store_txt_if_missing)


def produce_family_output(full_df: pd.DataFrame, family_names: list[str], family_tokenizers: list[PreTrainedTokenizerFast], country_name: str) -> list[Output]:
    ''' full_df is a df containing all the countries to test'''
    print(country_name, flush=True)

    family_gb = full_df.groupby('family')
    family_dfs = (family_gb.get_group(family) for family in family_names)

    test_df = full_df[full_df.country == country_name]

    results = [
        Output(
            (family_name, country_name),
            naive_alphabet_baseline(train_df, test_df),
            huffman_letter_baseline(train_df, test_df),
            huffman_name_baseline(train_df, test_df),
            evaluate_BPE(test_df.name, test_df.normed_count, fast_tokenizer),
            len(fast_tokenizer.get_vocab())
        )
        for family_name, train_df, fast_tokenizer in zip(family_names, family_dfs, family_tokenizers)
    ]
    print(country_name, 'done', flush=True)
    return results

def produce_country_output(train_df__group_name: tuple[pd.DataFrame, pd.DataFrame, str]) -> Output:
    train_df, group_name = train_df__group_name
    return Output(
        group_name,
        naive_alphabet_baseline(train_df, test_df=train_df),
        huffman_letter_baseline(train_df, test_df=train_df),
        huffman_name_baseline(train_df, test_df=train_df),
        *brute_force_best_BPE(group_name, train_df, test_df=train_df),
    )


def get_country_country_cost(full_df):
    ''' Gets the encoding cost (E[bits/word]) when training on a country and evaluating on the same country'''
    country_country_result_path = get_data_path('data/country_train_country_test_results.parquet')
    if country_country_result_path.exists():
        country_country_cost = pd.read_parquet(country_country_result_path)
    else:
        todo = [(df, name) for name, df in full_df.groupby('country')]
        with Pool(processes=6) as pool:
            country_country_cost = pd.DataFrame(map(produce_country_output, tqdm(todo)))
        country_country_cost.to_parquet(country_country_result_path)
    return country_country_cost

def try_train(task):
    try:
        return train_BPE_wrapper(task)
    except RuntimeError:
        return None

def get_family_country_cost(full_df, vocab_size: int):
    # Getting E[bits/name] when training BPE on a language family, then encoding on all countries (not just family)
    family_country_result_path = get_data_path(f'data/family_train_country_test_{vocab_size}_results.parquet')
    if family_country_result_path.exists():
        family_country_cost = pd.read_parquet(family_country_result_path)
    else:
        todo = [(name, df_to_name_seq(df), vocab_size) for name, df in full_df.groupby('family')]
        with Pool(processes=6) as pool: # 3 families, longest takes around 
            try:
                family_names, family_tokenizers = zip(*filter(None, pool.map(train_BPE_wrapper, todo)))
            except ValueError:
                return None
            if not len(family_tokenizers) == 3:
                return None
            todo = [
                (full_df, family_names, family_tokenizers, country_name) for country_name in full_df.country.unique()
            ]
            family_country_cost = pd.DataFrame([country_output for family_output in pool.starmap(produce_family_output, todo) for country_output in family_output])
        family_country_cost.to_parquet(family_country_result_path)
    return family_country_cost

def get_formatted_fcc(fcc, full_df):
    fcc = fcc.copy()
    fcc[['BPE_family', 'country']] = fcc.group_name.apply(pd.Series)
    fcc = (
        fcc.merge(full_df[['country', 'family']], on='country')
        .set_index(['BPE_family', 'country', 'family'])
        .drop(columns=['group_name', 'bpe_vocab'])
        .stack()
        .rename_axis(index={None: 'encoding'})
    )
    fcc = pd.DataFrame(fcc.to_list(), columns = ['bits_per_word', 'nan_percent'], index = fcc.index)
    return fcc.groupby(['BPE_family', 'family', 'encoding']).agg(SUMMARY)

def get_bpe_improvement_over_huff(fcc_formatted):
    '''
    Normalizes the name with respect to the default compression (ascii)
    Then returns the difference between BPE and huffman encoding
    '''
    baseline = fcc_formatted.xs('naive_letter', level='encoding')['bits_per_word']
    normalized = fcc_formatted['bits_per_word'] / baseline
    bpe = normalized.xs('bpe', level='encoding')
    huff_letter = normalized.xs('huff_letter', level='encoding')
    bpe_extra_compression = (huff_letter - bpe).squeeze().rename('Additional % compression of BPE compared to huffman letter encoding')
    return bpe_extra_compression.unstack().round(2).__mul__(100).astype(int)

def best_bpe_stats(fcc, full_df):
    fcc = pd.concat(filter(lambda x: x is not None, [get_family_country_cost(full_df, vocab_size) for vocab_size in [64, 128, 256, 512, 1024]]))
    fcc[['BPE_family', 'country']] = fcc.group_name.apply(pd.Series)
    fcc = fcc.merge(full_df[['country', 'family']].drop_duplicates(), on='country').set_index(['BPE_family', 'country', 'family', 'bpe_vocab']).bpe
    fcc = pd.DataFrame(fcc.to_list(), columns = ['bits_per_word', 'nan_percent'], index = fcc.index).bits_per_word
    mins = fcc.groupby(['BPE_family', 'country', 'family']).idxmin()
    bits_and_vocab_stats = fcc[mins].reset_index('bpe_vocab').groupby(['BPE_family', 'family']).agg({'bits_per_word': SUMMARY, 'bpe_vocab': SUMMARY}).loc[FAMILY_ORDER].round().astype(int)
    return bits_and_vocab_stats

def simple_example():
    full_df = just_give_me_all_the_shit('firstName')
    df = full_df[full_df.country == 'DK']
    print(f'{naive_alphabet_baseline(df, df) = }')
    print(f'{huffman_letter_baseline(df, df) = }')
    print(f'{huffman_name_baseline(df, df) = }')
    print(f'{brute_force_best_BPE(df, df) = }')

    res = produce_country_output((df, 'DK'))

if __name__ == '__main__':
    full_df = just_give_me_all_the_shit('firstName').query('family != "Non-Euro Romance/Germanic"')
    assert all(full_df.name.apply(len) <= 20) # to avoid running forever

    # Format and storage
    ccc = get_country_country_cost(full_df).set_index('group_name').drop(columns='bpe_vocab').stack().rename_axis(index={None: 'encoding'}).agg(pd.Series).set_axis(['bits_per_word', 'nan_percent'], axis=1)
    ccc_formatted = ccc.groupby('encoding').agg({'bits_per_word': SUMMARY})
    res_path = get_data_path('data/.txt')
    store_txt_if_missing(ccc_formatted, 'country_self_evaluation_bit_cost')

    fcc = get_family_country_cost(full_df, 512)
    if fcc is None:
        raise RuntimeError('too high vocab set?')
    fcc_formatted = get_formatted_fcc(fcc, full_df)
    bpe_vs_huff = get_bpe_improvement_over_huff(fcc_formatted)
    store_txt_if_missing(bpe_vs_huff, 'bpe_improvement_over_huff_results')

    stats = best_bpe_stats(fcc, full_df)
    store_txt_if_missing(stats, 'best_bpe_vocab_stats')

    seq = df_to_name_seq(just_give_me_all_the_shit('firstName'))
    train_BPE_wrapper(('full', seq, 4_096))

    result = ''
    for family in full_df.family.unique():
        fast_tokenizer = deserialize_fast_tokenizer(family, None)
        result += f'{family} {fast_tokenizer.vocab_size}\n'
    store_txt_if_missing(result, 'max_bpe_vocab')
