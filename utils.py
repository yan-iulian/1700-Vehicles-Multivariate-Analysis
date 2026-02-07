import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import numpy as np
import scipy.stats as stts

def replaceNan(df):
    for col in df:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(),inplace=True)
            else:
                df[col].fillna(df[col].mode()[0],inplace=True)



def toDataFrame(matrix,index,columns,file_name=None):
    data_frame=pd.DataFrame(matrix,index=index,columns=columns)
    if file_name is not None:
        data_frame.to_csv(file_name)
    return data_frame


def test_bartlett(r2,n,p,q,m):
    v=1-r2
    chi2=(-n+1+(p+q+1)/2)*np.log(np.flip(np.cumprod(np.flip(v))))
    n_lib=[ (p-k+1)*(q-k+1)
        for k in range(1,(m+1))
    ]
    p_values=1-stts.chi2.cdf(chi2,n_lib)
    return p_values

def histograma(x,variabila,partitie):
    clustere_unice=np.unique(partitie)
    fig,axs=plt.subplots(1,len(clustere_unice),figsize=(10,6),sharey=True)
    fig.suptitle(f"Histograme ale variabilei {variabila}")
    for ax,cluster in zip(axs,clustere_unice):
        ax.hist(x[partitie==cluster],bins=15,rwidth=0.9)
        ax.set_title(cluster)


