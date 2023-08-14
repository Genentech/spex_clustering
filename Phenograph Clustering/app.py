import phenograph
import anndata as ad
from anndata import AnnData
import scanpy as sc
from scipy.stats import zscore
from scipy.stats.mstats import winsorize

def phenograph_cluster(adata,knn,markers,transformation='arcsin', scaling='z-score'):
    
    expdf=adata.to_df() # Get this anndata from feature extraction output or spex load file input

    data_for_calc = expdf.iloc[:, markers] # marker selection in spex UI

    # Dropdown selection for transformation. Options are 'arcsin', 'log', 'none'
    if transformation=='arcsin':
        data_for_calc=np.arcsinh(data_for_calc/5)
    if transformation=='log':
        data_for_calc= data_for_calc.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x)

    # Dropdown selection for transformation. Options are 'z-score', 'winsorize', 'none'
    if scaling =='z-score':
        data_for_calc=data_for_calc.apply(zscore)
    if scaling =='winsorize':
        arrDataFrame_Winsorized = winsorize(data_for_calc.to_numpy(), limits=(0,.01)).data
        data_for_calc.iloc[:, :] = arrDataFrame_Winsorized

    # knn value selection from spex ui
    communities, graph, Q = phenograph.cluster(data_for_calc,k=knn,clustering_algo = 'leiden')

    bdata = ad.AnnData(data_for_calc)
    sc.pp.neighbors(bdata, n_neighbors=knn)
    sc.tl.umap(bdata,min_dist=0.5)
    adata.obsm['X_umap']=bdata.obsm['X_umap']

    adata.obs['cluster_phenograph'] = [str(i) for i in communities]

    return adata