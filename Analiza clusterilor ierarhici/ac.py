import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclust
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,silhouette_samples
from utils import *

masini=pd.read_csv('../SET DE DATE/Specificatii masini 2023.csv',index_col=(0,1))
replaceNan(masini)
variabile=masini.columns[2:].tolist()
observatii=masini.index.tolist()

n=len(observatii)   #marimea esantionului
p=n-1 #cati clusteri putem avea

x_orig=masini[variabile].values
x=StandardScaler().fit_transform(x_orig)

#crearea modelului de clusteri ierarhici
h=hclust.linkage(x,method='ward')
print(h)

#cautam nr optim de clusteri
k_max=np.argmax(h[1:,2]-h[:-1,2])
nr_clusteri=p-k_max
#nr_clusteri=7
print(f"Numarul optim de clusteri este: {nr_clusteri}")

#creare partitie automata cu fcluster
auto=fcluster(h,nr_clusteri,criterion='maxclust')

#indecsi Sillhouette
sill_instance=silhouette_samples(x,auto)  #ptr fiecare masina in parte
sill_partitie=silhouette_score(x,auto)  #media pe partitie
print(f"Media pe partitie este {np.round(sill_partitie,3)}")
toDataFrame(sill_instance.reshape(-1,1),masini.index,['Silhouette'],'Silhouette.csv')

#afisare la consola jonctiunea cu diferenta maxima si valoarea pragului
distante=h[:,2]
diferente=np.diff(distante)
pas_maxim=np.argmax(diferente)
valoare_max=distante[pas_maxim]
print(f"Jonctiunea (Pasul) cu valoarea maxima este {pas_maxim+1}")
print(f"Valoarea pragului de la jonctiunea maxima este: {valoare_max}")

#grafice

#desenare de histograme pentru primele 3 variabile
for i in range(3):
    histograma(x[:,i],variabile[i],auto)

#desenare dendrograma
plt.figure(figsize=(13,6))
plt.title('Dendrograma')
hclust.dendrogram(h,
                  no_labels=True,
                  leaf_rotation=90,
                  leaf_font_size=8)
plt.tight_layout()
plt.show()

