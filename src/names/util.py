from collections import Counter
import functools
import time
from pathlib import Path
from random import shuffle
from typing import Literal

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from unidecode import unidecode
from alphabet_detector import AlphabetDetector
import names as names_module

ALPHA_DETECTOR = AlphabetDetector()

ROOT = Path(names_module.__file__).parents[2]
DATA = ROOT / 'data'

SUMMARY = 'mean'
FAMILY_ORDER = [
    ('Euro Germanic', 'Euro Germanic'),
    ('Euro Romance', 'Euro Romance'),
    ('Other', 'Other'),
    ('Euro Germanic', 'Euro Romance'),
    ('Euro Germanic', 'Other'),
    ('Euro Romance', 'Euro Germanic'),
    ('Euro Romance', 'Other'),
    ('Other', 'Euro Germanic'),
    ('Other', 'Euro Romance'),
]

def get_language_corpus_words(language, number_of_lines=10000):
    """Takes one of the following language names: italian, danish, swedish, 
    portuguese, spanish and returns a list of tokenized lowercase words from corpus
    source: https://www.statmt.org/europarl/"""
    with open(DATA / f"corpus/{language}", encoding="utf-8") as f:
        language_lines = f.readlines()
    language_text = ""
    for idx, sentence in enumerate(language_lines):
        language_text += sentence
        if idx >= number_of_lines:
            break
    language_tokens = nltk.tokenize.word_tokenize(language_text, language=language, preserve_line=False)
    language_words = [unidecode(word.lower()) for word in language_tokens if word.isalpha()]
    return language_words


def shuffle_names(names):
    """Takes a list of names of various length and returns a dictionary with keys of each length and values of list_of_shuffled_names
    Preserves count of each length of names and shuffles characters between all names"""
    length_list = list({len(name) for name in names})
    length_list.sort()

    length_counts = Counter([len(name) for name in names])
    characters = list("".join(names))
    shuffle(characters)
    shuffled_name_dict = dict()
    for length in length_list:
        shuffled_names = []
        for _ in range(length_counts[length]):
            shuffled_name = ""
            for i in range(length):
                shuffled_name += characters.pop()
            shuffled_names.append(shuffled_name)
        shuffled_name_dict[length] = shuffled_names

    return shuffled_name_dict

def translate(trans_dict, names):
    """translates latters in names depending on trans_dict then return list of translated names"""
    translated_names = []
    for name in names:
        for letter in name:
            if letter in trans_dict.keys():
                new_name = name.replace(letter,trans_dict[letter])
        translated_names.append(new_name)
    return translated_names

def timeit(func: callable):
    name = func.__name__

    functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        res = func(*args, **kwargs)
        print(f'Running time of {name}:', time.monotonic() - start)
        return res
    return wrapper
    
def printing_tqdm(seq: iter, print_idx: int = None):
    ''' TQDM doesnt work in threads, so do this '''
    for val in seq:
        to_print = val if print_idx is None else val[print_idx]
        start = time.monotonic()
        print('starting', to_print, end=' ', flush=True)
        yield val
        print('ended', to_print, int(time.monotonic()-start), flush=True, end=' ')

def store_txt_if_missing(df, filename):
    assert 'data/' not in filename and '.' not in filename and 'results/' not in filename
    res_path = get_data_path(f'data/results/{filename}.txt')
    if not res_path.exists():
        with open(res_path, 'w') as f:
            f.write(df.to_string() if not isinstance(df, str) else df)
    
# def get_names_df(n=1000, name=True, countries=['FI', 'HR', 'IT', 'GB', 'DK', 'FR']):
#     if name:
#         name='name'
#     else:
#         name='surname'
        

#     with open(DATA / f'{name}_vectorizer.pickle', 'rb') as f:
#         vectorizer = pickle.load(f)
#     with open(DATA / f'{name}_matrix.pickle', 'rb') as f:
#         X = pickle.load(f)
#     name_array = vectorizer.get_feature_names_out()
#     df = pd.DataFrame(X.A)
    
#     country2id = dict()
#     id2country = dict()
    
#     for i, file in enumerate(DATA.glob('raw_country_names/*.parquet')):
#         t = file.stem
#         country2id[t] = i
#     country_ids = [country2id[country] for country in countries]
#     for country in countries:
#         id2country[country2id[country]] = country
        
#     def name_mapper(m): return name_array[int(m)]
    
#     dfCountries = df[df.index.isin(country_ids)]
#     dfCountries = dfCountries.loc[:, (dfCountries != 0).any(axis=0)]
    
#     country_series = []
    
#     for id in country_ids:
#         tempdf = dfCountries.loc[id].sort_values(ascending=False)
#         tempdf = tempdf.rename(index=name_mapper)
#         country_series.append(tempdf.iloc[:n].reset_index().rename(columns={'index':id2country[id]})[id2country[id]])
#     names_df = pd.concat(country_series,axis=1)
#     names_df = ( # clean
#         names_df.applymap(unidecode)
#         .applymap(str.lower)
#         .apply(lambda x: x.str.replace('[^a-z\-]', '', regex=True))
#     )
    
#     return names_df

def flatten_names_df(names_df) -> pd.Series:
    return names_df.rename_axis(columns='country').stack().reset_index(0, drop=True).sort_index()


def get_names_by_freq(name_col=0, n_names=10_000):
    corpus = []
    for file in tqdm(DATA.glob('raw_country_names/*.parquet')):
        df = pd.read_parquet(file, columns=[name_col])
        counts = df[name_col].value_counts()[:10000].reset_index()
        counts['index'] = counts['index'].str.replace(" ", "-")
        temp = (counts['index']+" ").str.repeat(counts[name_col])
        corpus.append(temp.to_frame().T.apply(" ".join, axis=1))
    return corpus

def path_to_name_counts(path: Path, column: str = 'firstName', n_names: int = 10_000):
    ''' returns df with columns [name	count	country]'''
    df = pd.read_parquet(path, columns=[column])
    counts = df[column].value_counts()[:n_names]
    formatted = counts.rename_axis('name').rename('count').reset_index().assign(country=path.stem)
    return formatted

def _get_country_name_counts(column):
    cache = DATA / f'{column}_country_counts.parquet'
    if cache.exists():
        return pd.read_parquet(cache)

    files = tqdm(list(DATA.glob('raw_country_names/*.parquet')))
    try:
        df = pd.concat((path_to_name_counts(f, column=column) for f in files)) # Must use () and not [] for memory
    except ValueError as e:
        raise RuntimeError('No cache and missing raw files') from e
    df.to_parquet(cache)
    return df

def clean_country_name_counts(column: Literal['firstName'] | Literal['lastName']):
    ''' After cleaning, some names might be duplicate. Add counts'''
    assert column in ['firstName', 'lastName']
    MIN_UNIQUE_NAMES_AFTER_CLEANING = 2000
    MIN_LATIN_NAMES_RATIO = .99
    MIN_COUNT_TO_INCLUDE = 3

    df = _get_country_name_counts(column)
    df = (
        df.groupby('country')
        .filter(lambda x: (x['name'].apply(ALPHA_DETECTOR.is_latin) * x['count']).sum()/x['count'].sum() > MIN_LATIN_NAMES_RATIO)
    )
    
    df['name'] = (
        df['name']
        .str.lower()
        .str.strip()
        .str.replace(r'[^a-z äëïöüßâêôåáéíóúàèìòùãõñç]', '', regex=True)
        .str.replace(r'\s+', '-', regex=True)
    )
    df = df.groupby(['country', 'name']).sum().reset_index() # merge names that become identical after normalizing text
    df = df.query(f'count > {MIN_COUNT_TO_INCLUDE}') # remove e.g. typos and concatenated names

    df = df.groupby('country').filter(lambda x: x['name'].count() > MIN_UNIQUE_NAMES_AFTER_CLEANING)
    return df

def get_n_tfidf_names(df: pd.DataFrame, n_top_names: int = 1000):
    ''' df = clean_country_name_counts(column) '''
    count_matrix = df.set_index(['country', 'name']).unstack('name').fillna(0)

    tfidf = pd.DataFrame(
        TfidfTransformer().fit_transform(count_matrix).toarray(), 
        columns=count_matrix.columns,
        index=count_matrix.index
    ).rename(columns={'count': 'tfidf'})

    n_largest_ = tfidf.stack().groupby('country').apply(lambda x: x.nlargest(n_top_names, columns='tfidf'))
    assert list(n_largest_.index.names) == ['country', 'country', 'name']
    n_largest = n_largest_.droplevel(0)
    assert list(n_largest.index.names) == ['country', 'name']

    assert (n_largest != 0).all().all()

    return n_largest

def add_freq_to_tfidf(country_name_counts: pd.DataFrame, top: pd.DataFrame):
    '''
    col = 'firstName'
    country_name_counts = clean_country_name_counts(column=col)
    top = get_n_tfidf_names(country_name_counts, n_top_names=1000)

    Normalize ensures that the sum of counts within country is approximately equal
    '''
    count_map = country_name_counts.set_index(['country', 'name'])['count'].to_dict()
    top_w_freq = top.assign(freq = top.index.map(count_map)) # use .assign(...) to create a copy
    top_w_freq['freq_percent'] = top_w_freq.groupby('country', group_keys=False)['freq'].apply(lambda x: x/x.sum())
    return top_w_freq

def get_normed_counts(top_w_count, SCALE = 10):
    '''
    col = 'firstName'
    country_name_counts = clean_country_name_counts(column=col)
    top = get_n_tfidf_names(country_name_counts, n_top_names=1000)
    top_w_count = add_freq_to_tfidf(country_name_counts, top)
    
    SCALE = 10 # Increasing this reduces rounding errors, but increases running time
    '''
    normed_count = (top_w_count.freq_percent * SCALE / top_w_count.freq_percent.min()).round().astype(int)

    assert (normed_count != 0).all()

    stats = normed_count.groupby('country').sum().describe()
    assert (stats.loc['max'] - stats.loc['min']) / SCALE < 50 # There's at most a difference of 50 counts away from each country being equal in cumsum

    return normed_count.rename('normed_count')

def get_language_families():
    romance = set(["CO", "CL", "PE", "BO", "MX", "GT", "EC", "SV", "HN", "PA", "PR", "CR", "UY", "AR", "ES", "IT", "AO", "PT", "FR", "BE", "BR", "CM", "MD", "LU"])
    germanic = set(["SE", "DK", "NO", "IS", 'GB', 'DE', 'NL', 'AT', 'IE', 'CH', 'US', 'AT', 'CA', 'AU', 'NZ', 'JM'])
    euro_romance = set(['ES', 'PT', 'FR', 'IT'])
    euro_germanic = set(['GB', 'SE', 'DE', 'NL'])
    L = []
    with open(f"{DATA}/country_codes.txt", 'r') as f:
        for country in map(str.strip, f):
            if country in euro_romance:
                L.append((country,"Euro Romance"))
            elif country in euro_germanic:
                L.append((country,"Euro Germanic"))
            elif country in romance or country in germanic:
                L.append((country,"Non-Euro Romance/Germanic"))
            else:
                L.append((country,"Other"))
                
        return pd.DataFrame(L, columns =["country","family"])

def just_give_me_all_the_shit(col):
    ''' Remember to inc version if modify '''
    version = 6
    cache = DATA / f'{col}_data_cache_{version}.parquet'
    if cache.exists():
        return pd.read_parquet(cache)

    country_name_counts = clean_country_name_counts(column=col)
    top = get_n_tfidf_names(country_name_counts, n_top_names=1000)
    top_w_count = add_freq_to_tfidf(country_name_counts, top)
    assert top_w_count.groupby('country').count().min().min() == 1000

    normed_counts = get_normed_counts(top_w_count)
    all_counts = pd.concat([top_w_count, normed_counts], axis=1)
    all_data = all_counts.reset_index('name').merge(get_language_families(), on='country')
    all_data.to_parquet(cache)
    return all_data

def get_data_path(path: str) -> Path:
    assert 'data/' in path, 'must prefix path with "data/"'
    return ROOT / path

if __name__ == '__main__':
    col = 'firstName'
    all_data = just_give_me_all_the_shit(col)
    ((all_data.set_index('family').name.agg(len)*all_data.normed_count.values).groupby('family').sum()/all_data.groupby('family').normed_count.sum()).round(1)
    all_data['len'] = all_data.name.agg(len)
    all_data.sort_values('len').query('family in ["Euro Germanic", "Euro Romance"]').head(50)