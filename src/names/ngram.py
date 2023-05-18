
import functools
import pandas as pd
import itertools
#from names.util import get_names_df, flatten_names_df
import holoviews as hv
import hvplot.pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from typing import NamedTuple
hv.extension('bokeh')

def get_ngrams(names: pd.Series, n: int):
    '''
    Pads names so they all have the same size
    Start-of-word: ^
    End-of-word: $
    '''
    padded_names = '^'*(n-1) + names + '$'*(n-1)
    longest_length = padded_names.apply(len).max()
    all_names: str = ''.join(padded_names.str.ljust(longest_length+1))

    char_series = pd.Series(list(all_names))
    n_grams = pd.concat([
        char_series.shift(-i).rename(i) for i in range(n)
    ], axis=1).dropna().apply(''.join, axis=1)

    grams_w_padding = n_grams.str.contains(' ')
    valid_n_grams = n_grams[~grams_w_padding]
    return valid_n_grams

@functools.cache
def get_possible_ngrams(n):
    chars = 'abcdefghijklmnopqrstuvwxyz-^$'
    possible_ngrams = pd.Series(map(''.join, itertools.product(*([chars]*n))))
    return possible_ngrams

def get_countries_ngrams(all_names: pd.Series, n: int) -> pd.Series:
    ''' all_names = names.utils.flatten_names_df(...) '''
    if n == 1:
        return all_names.apply(list).groupby('country').agg(list)

    all_ngrams = get_ngrams(all_names, n)

    name_starts = all_ngrams.str.contains('^'*(n-1), regex=False)
    name_ids = name_starts.astype(int).cumsum()
    ngram_series = all_ngrams.groupby(name_ids).agg(list).groupby(all_names.index).agg(list)

    return ngram_series

# def countries_to_df(countries_ngrams: pd.Series) -> pd.Series:
#     return countries_ngrams.agg(pd.Series).stack().agg(pd.Series).unstack()

def get_ngram_probs(all_names: pd.Series, n: int, laplace=True):
    n_grams = all_names.groupby('country').agg(get_ngrams, n).apply(pd.Series)

    if laplace:
        possible_ngrams = get_possible_ngrams(n)
        possible_ngrams_rep = pd.concat([possible_ngrams]*len(n_grams), axis=1).T
        possible_ngrams_rep.index = n_grams.index

        n_grams = pd.concat([n_grams, possible_ngrams_rep], axis=1)

    probs = n_grams.stack().groupby('country').value_counts(normalize=True).rename_axis(index={None: 'ngram'})
    return probs

def get_bag_of_word_features(countries_ngrams: pd.Series) -> pd.Series:
    '''
    rows: country
    column: word
    cell: binary BoW feature vector of length "possible_ngrams
    '''
    possible_ngrams = get_possible_ngrams(NGRAM)
    features = countries_ngrams.apply(pd.Series).applymap(possible_ngrams.isin).applymap(lambda x: x.values)
    return features

def get_feature_df(bow_features) -> pd.DataFrame:
    feature_df = bow_features.stack().apply(pd.Series)
    feature_df.columns = get_possible_ngrams(NGRAM)
    return feature_df

def plot_dendrogram(model, **kwargs):
    from scipy.cluster.hierarchy import dendrogram
    # from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    
def get_kmeans_countries( k, ngram=3, style='matplotlib'):
    import os
    from pathlib import Path
    import names as names
    root = Path(names.__file__).parents[2]
    countries = [p.stem for p in root.glob('data/*')]
    names_df = get_names_df(n=1000, countries=countries)
    all_names = flatten_names_df(names_df)
    countries_ngrams = get_countries_ngrams(all_names, ngram)
    flat_ngrams = countries_ngrams.apply(sum, start=[])
    existing_ngrams = set(flat_ngrams.sum())
    ngram_ratio = flat_ngrams.apply(pd.Series).stack().groupby('country').value_counts(normalize=True)
    idx = pd.MultiIndex.from_product([countries, existing_ngrams], names=['country', 'gram'])
    data = ngram_ratio.reindex(idx).fillna(0).unstack()
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    model = model.fit(data)
    return data, model

    # # N_COMPONENTS = 40
    # # cols = {'columns': [f'PC{i}' for i in range(N_COMPONENTS)]}
    # # pca = PCA(n_components=N_COMPONENTS)
    # # data_T = pd.DataFrame(pca.fit_transform(data), **cols, index=data.index)
    # # centers_T = pd.DataFrame(pca.transform(k_means_centers), **cols)
    
    # k_means = KMeans(n_clusters=k).fit(data)
    # k_means_centers = k_means.cluster_centers_
    # k_means.labels_
    # name_scatter.opts(tick_position='top')

    # hv.extension(style)
    # match style:
    #     case 'matplotlib':
    #         name_scatter = hv.Scatter3D(data_T).opts(alpha=.1, cmap=data.index)
    #         center_scatter = hv.Scatter3D(centers_T).opts(s=1000)
    #     case 'bokeh':
    #         axes = dict(kdims=data_T.columns.to_list())
    #         name_scatter = hv.Points(data_T.reset_index(), **axes).opts(alpha=.1, tools=['hover'])
    #         center_scatter = hv.Points(centers_T).opts(size=10)
    # name_scatter * center_scatter

def get_kmeans_countries_names(countries_ngrams, k, style='matplotlib'):
    bow_features = get_bag_of_word_features(countries_ngrams)
    data = get_feature_df(bow_features)

    k_means_centers = KMeans(n_clusters=k).fit(data).cluster_centers_

    N_COMPONENTS = 2 if style == 'bokeh' else 3
    cols = {'columns': [f'PC{i}' for i in range(N_COMPONENTS)]}
    pca = PCA(n_components=N_COMPONENTS)
    data_T = pd.DataFrame(pca.fit_transform(data), **cols)
    centers_T = pd.DataFrame(pca.transform(k_means_centers), **cols)
    name_scatter = hv.Scatter3D(data_T).opts(alpha=.1, cmap=data.index)

    hv.extension(style)
    match style:
        case 'matplotlib':
            center_scatter = hv.Scatter3D(centers_T).opts(s=1000)
        case 'bokeh':
            center_scatter = hv.Points(centers_T).opts(size=10)
    return name_scatter * center_scatter

def get_kmeans_assignments(countries_ngrams):
    bow_features = get_bag_of_word_features(countries_ngrams)
    data = get_feature_df(bow_features)

    clf = KMeans(n_clusters=5).fit(data)

    countries = data.reset_index(level=1, drop=True).index
    assignments = pd.DataFrame(zip(countries, clf.labels_), columns=['country', 'cluster'])
    return assignments.groupby('cluster').value_counts()

def country_grams_to_probs(grams):
    ''' grams comes from get_counties_ngrams '''
    return (
        grams.apply(sum, args=([],))
        .apply(pd.Series).stack().rename('grams')
        .groupby('country')
        .value_counts(normalize=True)
        .rename('prob')
    )

def calculate_probabilities(all_names, n_gram):
    class Probs(NamedTuple):
        p_of_l: pd.Series # P(w_i)
        p_of_prev: pd.Series # P(w_i-1)
        # The above two should be identical except for the chars ^ and $
        p_of_l_and_prev: pd.Series # P(w_i, w_i-1)
        p_of_l_given_prev: pd.Series # P(w_i|w_i-1)
    
    if n_gram < 2:
        raise NotImplementedError('Cannot do conditionals when n < 2')
    
    p_gram = country_grams_to_probs(get_countries_ngrams(all_names, n_gram)).reset_index()
    p_gram['prev'] = p_gram.grams.str[:-1]
    p_gram['now'] = p_gram.grams.str[-1]

    p_of_l_and_prev = p_gram.set_index(['country', 'prev', 'now']).prob
    assert (p_of_l_and_prev.groupby('country').sum() == 1).all()

    # p(letter at idx=i)
    p_of_l = p_of_l_and_prev.groupby(['country', 'now']).sum()
    assert (p_of_l.groupby('country').sum() == 1).all()

    # p(letter(s) at idx=i-1 (, i-2...))
    p_of_prev = p_of_l_and_prev.groupby(['country', 'prev']).sum()
    assert (p_of_prev.groupby('country').sum() == 1).all()

    p_of_l_given_prev = p_of_l_and_prev / p_of_prev.reindex(p_of_l_and_prev.index).sort_index()
    assert (p_of_l_given_prev.groupby(['country', 'prev']).sum().round(9) == 1).all()

    return Probs(
        p_of_l = p_of_l,
        p_of_prev=p_of_prev,
        p_of_l_and_prev=p_of_l_and_prev,
        p_of_l_given_prev=p_of_l_given_prev,
    )
    
def entropy(p: pd.Series):
    return (-p*np.log2(p)).groupby('country').sum()

def kullback_leibler(p_x_y, p_x, p_y):
    return (p_x_y*np.log2(p_x_y/(p_x*p_y))).groupby('country').sum()

def conditional_entropy(p_x_y, p_x):
    H = -p_x_y*np.log2(p_x_y/p_x)
    H[np.isinf(H)] = 0
    return H.groupby('country').sum()

def print_entropy_stats(probs: pd.Series) -> None:
    ''' probs = calculate_probabilities(all_names, NGRAM) '''
    p_of_l, p_of_prev, p_of_l_and_prev = map(probs.__getattribute__, ['p_of_l', 'p_of_prev', 'p_of_l_and_prev'])

    p_x_y = p_of_l_and_prev.sort_index()
    # We need to swap levels for broadcasting to be correctly applied
    p_y = p_of_l.reindex(p_x_y.index.swaplevel()).sort_index().swaplevel()
    p_x = p_of_prev.reindex(p_x_y.index).sort_index()

    # Mutual information
    print(f'{kullback_leibler(p_x_y, p_x, p_y) = }')
    # IMPORTANT: don't use p_x here, since it is broadcastet and therefore gives higher and misleading entropy
    print(f'{entropy(p_of_l) + entropy(p_of_prev) - entropy(p_x_y) = }')

    # Conditional entropy
    print(f'{conditional_entropy(p_x_y, p_x) = }')
    # IMPORTANT: don't use p_x here, since it is broadcastet and therefore gives higher and misleading entropy
    print(f'{entropy(p_x_y) - entropy(p_of_prev)}')


if __name__ == '__main__':
    names_df = get_names_df(n=100)
    all_names = flatten_names_df(names_df)

    NGRAM = 2
    N_COUNTRIES = len(names_df.columns)
    assert N_COUNTRIES < 100
    probs = calculate_probabilities(all_names, NGRAM)
    print_entropy_stats(probs)

    # Ideas
    # examine which countries have similar tri-gram distributions
    # examine shared -grams between countries

    # gb = p_of_l_and_prev.groupby('country')
    # for (c1, df1), (c2, df2) in itertools.product(gb, gb):
    #     if c2 == 'DK':
    #         continue
    #     break

    # df1
    # df2

def get_gram_distributions(all_names, ngram):
    grams = get_countries_ngrams(all_names, 1)
    countries = df.country.unique()
    df = grams.reset_index()
    # The following can un-share axes and sort by greatest value
    # bars = hv.Bars(p_uni, kdims=['country', 'grams'])
    # hv.Layout(list(map(bars.__getitem__, countries))).opts(shared_axes=False).cols(3)
    # for c in countries:
    #     print(c)
    #     bars[c]
    return df.hvplot.bar('country', by='grams')

# def examine_largest_contributing_features():
#     largest_idxs = pca.components_[0].argsort()[::-1][:30]

#     data = X.iloc[:, largest_idxs].astype(int)
#     #data.columns = data.columns.str.replace('$', 'E')
#     print(data.reset_index().drop(columns='level_1').groupby(data.columns.to_list()).value_counts().to_string())

#     all_names[all_names.str[-3:] == 'ana']

#     all_names[all_names.str.contains('nna')]
#     pca.components_[0].argmax()

#     sum(pca.explained_variance_)
