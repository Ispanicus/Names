import functools
import itertools
from multiprocessing.pool import Pool
from typing import NamedTuple

import country_converter as coco
import numpy as np
import pandas as pd

from names.bpe import deserialize_fast_tokenizer
from names.util import (FAMILY_ORDER, SUMMARY, get_data_path,
                        just_give_me_all_the_shit, store_txt_if_missing)


class Distribution(NamedTuple):
    group_name: str
    vocab_family_name: str
    counts: dict[str: int]

def get_vocab_counts(name, df, vocab_family_name, vocab):
    print(name, 'started', flush=True)
    counts = {word: (df.name.str.count(word) * df.normed_count).sum() for word in vocab}
    print(name, 'done', flush=True)
    return Distribution(name, vocab_family_name, counts)

def build_dists(full_df):
    cache_path = get_data_path('data/dists.parquet')
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    
    family_tokenizers = {family_name: deserialize_fast_tokenizer(family_name, 512) for family_name in full_df.family.unique()}

    todo = [(name, df, vocab_family_name, tokenizer.get_vocab())
            for vocab_family_name, tokenizer in family_tokenizers.items()
            for name, df in itertools.chain(full_df.groupby('family'), full_df.groupby('country'))]
    with Pool(processes=6) as pool: # 3 families, longest takes around 
        dists = pd.DataFrame(pool.starmap(get_vocab_counts, todo))
    dists.to_parquet(cache_path)
    return dists

def get_dist(family: str):
    full_df = just_give_me_all_the_shit('firstName')
    dists = build_dists(full_df)
    counts = dists.set_index(['group_name', 'vocab_family_name']).xs(family, level='vocab_family_name').squeeze().agg(pd.Series).dropna(axis=1)
    prob = counts.div(counts.sum(axis=1), axis=0)
    return prob

def clean_dist(dists: pd.DataFrame):
    return dists[dists != 0].dropna(how='all', axis=1)

def kl_divergence(P: pd.Series, Q: pd.Series):
    assert (P.index == Q.index).all(), 'must align indices'
    assert (P[(Q == 0)] == 0).all().all(), 'Q(x)=0 implies P(x)=0'
    return (P * (P / Q).apply(np.log2)).sum(skipna=False)

def get_avg_kl(dists, family_name):
    dists = clean_dist(dists)
    target = dists.xs(family_name).squeeze()
    kl_func = functools.partial(kl_divergence, Q = target)
    kl = dists.apply(kl_func, axis=1).rename('kl')
    return kl

def ordered_kl_divergence(pair_probs: pd.Series, P_country_names: list[str], Q_family_name: str):
    dists_ = clean_dist(pair_probs.xs(Q_family_name, level=1))
    Q = dists_.xs(Q_family_name).squeeze()
    dfs = []
    for country in P_country_names:
        P = dists_.xs(country).squeeze()
        assert (P.index == Q.index).all(), 'must align indices'
        assert (P[(Q == 0)] == 0).all().all(), 'Q(x)=0 implies P(x)=0'
        dfs.append((P * (P / Q).apply(np.log2)).rename(country))#.sort_values(ascending=False).head(5).to_frame().T)
    #return pd.concat(dfs, axis=1).mean(axis=1).sort_values(ascending=False).head(5).to_frame().T
    return pd.concat(dfs, axis=1).mean(axis=1).sort_values(ascending=False).to_frame().T

if __name__ == '__main__':
    full_df = just_give_me_all_the_shit('firstName').query('family != "Non-Euro Romance/Germanic"')
    dists = build_dists(full_df)

    counts = dists.set_index(['group_name', 'vocab_family_name']).squeeze().agg(pd.Series)
    counts += 1 # Laplace
    non_letters = (counts.columns.to_series().apply(len) > 1).values
    pair_counts = counts.loc[:, non_letters][lambda x: x.columns.difference(full_df.name)]
    pair_probs = pair_counts.div(counts.sum(axis=1), axis=0)

    kl = pair_probs.groupby('vocab_family_name').apply(lambda df: get_avg_kl(df, df.name)).reset_index(0, drop=True).rename_axis(index={'vocab_family_name': 'target_dist'})
    kl_w_family = full_df[['country', 'family']].drop_duplicates().merge(kl.rename_axis(index={'group_name': 'country'}).reset_index(), on='country')
    kl_summary = (
        kl_w_family
        .groupby(['target_dist', 'family'])
        .agg({'kl': SUMMARY})
        .loc[FAMILY_ORDER]
    )
    kl_piv_summary = kl_summary.set_axis(['mean kl'], axis=1).unstack()
    store_txt_if_missing(kl_piv_summary.round(2), 'kl_summary')

    # deviations
    df = kl_w_family[kl_w_family.family == kl_w_family.target_dist]
    cc = {country: coco.convert(country, to='name_short') for country in df.country.unique()}
    worst_offenders = ''
    for name, target_dist in df.groupby('target_dist'):
        target_dist = target_dist.set_index('country').rename(index=cc)
        ordered = target_dist.kl.sort_values(ascending=False).head(10).rename_axis(f'KL from {name}')
        worst_offenders += ordered.round(2).to_string() + '\n'
    kl_w_family.query('target_dist == "Euro Romance" and family == "Euro Germanic"')
    kl_w_family.query('target_dist == "Euro Germanic" and family == "Euro Romance"')
    store_txt_if_missing(worst_offenders, 'kl_contributions')
    print(worst_offenders)

    target_name = 'Euro Germanic'
    target = pair_probs.xs(target_name).xs(target_name)
    FR = pair_probs.xs('FR').xs(target_name)

    contributions = FR * (FR / target).agg('log2')
    contributions.sort_values().head(25).rename_axis('FR_vs_euro_romance_contributions').to_string()
    contributions.sort_values(ascending=False).head(50).rename_axis('FR_vs_euro_romance_contributions')
    (FR*(FR/target).agg('log2')).sum()

    full_df[full_df.name.str.contains('rie')].query('country == "FR"').sort_values('freq')
    full_df[full_df.name.str.contains('christ')].query('country == "FR"').sort_values('freq')

    euro_romance = set(['ES', 'PT', 'FR', 'IT'])
    euro_germanic = set(['GB', 'SE', 'DE', 'NL'])

    df1 = ordered_kl_divergence(pair_probs, euro_germanic, "Euro Romance")
    df2 = ordered_kl_divergence(pair_probs, euro_romance, "Euro Germanic")

    dfs = []
    for Q_family_name in ["Euro Romance", "Euro Germanic"]:
        dists_ = clean_dist(pair_probs.xs(Q_family_name, level=1))
        Q = dists_.xs(Q_family_name).squeeze()
        P = dists_.xs('FR').squeeze()
        assert (P.index == Q.index).all(), 'must align indices'
        assert (P[(Q == 0)] == 0).all().all(), 'Q(x)=0 implies P(x)=0'
        dfs.append((P * (P / Q).apply(np.log2)).rename(country))#.sort_values(ascending=False).head(5).to_frame().T)
    pd.concat(dfs, axis=1).stack().sort_values(reverse=True)

    df1
    full_df[full_df.name.str.contains('ke')].query('family == "Euro Germanic"').sort_values('normed_count').tail(50)
    full_df[full_df.name.str.contains('ke')].query('family == "Euro Romance"').sort_values('normed_count').tail(50)
    full_df[full_df.name.str.contains('ke')].query('family != "Other"').sort_values('normed_count').tail(50)

    df2
    full_df[full_df.name.str.contains('gi')].query('family == "Euro Romance"').sort_values('normed_count').tail(50)
    full_df[full_df.name.str.contains('gi')].query('country == "IT"').sort_values('freq').freq_percent.count()
    full_df[full_df.name.str.contains('gi')].query('family == "Euro Germanic"').sort_values('normed_count').tail(50)

    full_df[full_df.name.str.contains('io')].query('family == "Euro Romance"').sort_values('normed_count').tail(50)
    full_df[full_df.name.str.contains('io')].query('family == "Euro Romance"').sort_values('normed_count')[lambda x: x.name.str[-2:] == 'io'].groupby('country').freq_percent.sum()
    full_df[full_df.name.str.contains('io')].query('family == "Euro Germanic"').sort_values('normed_count').tail(50)

    # Most common in each language
    for col in pd.concat([df1, df2]):
        (
            full_df[full_df.name.str.contains(col)]
            .query('family.str.contains("Euro")')
            .sort_values('normed_count', ascending=False)
            .head(10).set_index(['family', 'country'])[["name", "freq_percent"]]
        )

    deserialize_fast_tokenizer('SE', None).vocab_size
    deserialize_fast_tokenizer('GB', None).vocab_size
    deserialize_fast_tokenizer('DE', None).vocab_size
    deserialize_fast_tokenizer('NL', None).vocab_size
    deserialize_fast_tokenizer('FR', None).vocab_size
    deserialize_fast_tokenizer('ES', None).vocab_size
    deserialize_fast_tokenizer('PT', None).vocab_size
    v = deserialize_fast_tokenizer('IT', 512).vocab
    sorted(v, key=v.__getitem__)

    