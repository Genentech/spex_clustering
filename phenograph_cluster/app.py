import phenograph
import anndata as ad
import re
from anndata import AnnData
import scanpy as sc
from scipy.stats import zscore
from scipy.stats.mstats import winsorize
import numpy as np


def phenograph_cluster(
        adata,
        knn,
        markers,
        transformation='arcsin',
        scaling='z-score'
):

    expdf = adata.to_df()  # Get this anndata from feature extraction output or spex load file input

    data_for_calc = expdf.iloc[:, markers]  # marker selection in spex UI

    # Dropdown selection for transformation. Options are 'arcsin', 'log', 'none'
    if transformation == 'arcsin':
        data_for_calc = np.arcsinh(data_for_calc/5)
    if transformation == 'log':
        data_for_calc = data_for_calc.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)

    # Dropdown selection for scaling. Options are 'z-score', 'winsorize', 'none'
    if scaling =='z-score':
        data_for_calc = data_for_calc.apply(zscore)
    if scaling =='winsorize':
        arr_data_frame_winsorized = winsorize(data_for_calc.to_numpy(), limits=(0, .01)).data
        data_for_calc.iloc[:, :] = arr_data_frame_winsorized

    # knn value selection from spex ui
    communities, graph, Q = phenograph.cluster(
        data_for_calc.values.tolist(),
        k=knn,
        clustering_algo='leiden'
    )

    bdata = ad.AnnData(data_for_calc)
    sc.pp.neighbors(bdata, n_neighbors=knn)
    sc.tl.umap(bdata, min_dist=0.5)
    adata.obsm['X_umap'] = bdata.obsm['X_umap']

    adata.obs['cluster_phenograph'] = [str(i) for i in communities]

    return adata


def run(**kwargs):

    adata = kwargs.get('adata')
    scaling = kwargs.get('scaling')
    transformation = kwargs.get('transformation')
    knn = kwargs.get('knn')
    channel_list = kwargs.get("channel_list", [])
    channel_list = [
        re.sub("[^0-9a-zA-Z]", "", item).lower().replace("target", "") for item in channel_list
    ]
    all_channels = kwargs.get("all_channels", [])

    if len(channel_list) > 0:
        channel_list: list[int] = [
            all_channels.index(channel)
            for channel in channel_list
            if channel in all_channels
        ]
    else:
        channel_list = list(range(len(all_channels)))

    adata = phenograph_cluster(
        adata,
        knn,
        markers=channel_list,
        scaling=scaling,
        transformation=transformation
    )

    return {'adata': adata}
