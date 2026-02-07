import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sb

from utils import toDataFrame,replaceNan

masini=pd.read_csv('../SET DE DATE/Specificatii masini 2023.csv',index_col=(0,1))
replaceNan(masini)
variabile_numerice=masini.columns[2:].tolist()
x_orig=masini[variabile_numerice].values
m,n=x_orig.shape

#standardizare
x=StandardScaler().fit_transform(x_orig)

#crearea modelului
model_acp=PCA()
model_acp.fit(x)

#valorile proprii ale componentelor
alpha=model_acp.explained_variance_
print(alpha)

#vectorii proprii ai modelului
a=model_acp.components_

#componentele principale
eticheteComponente=['C'+str(i+1) for i in range(len(alpha))]
c=model_acp.transform(x)
cDf=toDataFrame(c,masini.index,eticheteComponente,'Componente.csv')

#corelatia dintre variabilele initiale si componentele gasite
r_x_c=np.corrcoef(x,c,rowvar=False)[:n,n:]
rxcDf=toDataFrame(r_x_c,variabile_numerice,eticheteComponente,'Corelatii.csv')

#comunalitati
r_patrat=r_x_c**2
comunalitati=np.cumsum(r_patrat,axis=1)
comunalitatiDf=toDataFrame(comunalitati,variabile_numerice,eticheteComponente,'Comunalitati.csv')

#cosinusuri
c_patrat=c**2
sume=np.sum(c_patrat,axis=1,keepdims=True)
cosinusuri=c_patrat/sume
cosinusuriDf=toDataFrame(cosinusuri,masini.index,eticheteComponente,'Cosinusuri.csv')

#contributii
contributii=c_patrat/(m*alpha)
contributiiDf=toDataFrame(contributii,masini.index,eticheteComponente,'Contributii.csv')

#criterii de alegere a componentelor rezultate

#criteriul Kaiser - alegem doar componentele mai mari ca 1
conditie=np.where(alpha>1)
print(f'Cu criteriul Kaiser, sunt necesare doar primele {len(conditie[0])} componente')

#criteriul Cattell - cautam unde graficul componentelor (Scree plot) face o cotitura
eps=alpha[0:(n-1)]-alpha[1:n]
sig=eps[0:(n-2)]-eps[1:(n-1)]
conditie=np.where(sig<0)
print(f'Cu criteriul Cattell, sunt necesare doar primele {conditie[0][0]+1} componente')

#criteriul care pastreaza primele n componente care explica x% variatie, x=80%
ponderi=np.cumsum(alpha/sum(alpha))
conditie=np.where(ponderi>0.8)
print(f'Pentru a explica 80% din variatia variabilelelor initiale sunt necesare primele {conditie[0][0]+1} componente')

#grafice

#graficul componentelor C1 si C2 in raport cu observatiile setului de date
plt.figure(figsize=(10, 7))
plt.scatter(cDf['C1'], cDf['C2'], alpha=0.5, color='blue')
plt.title('Plotul Componentelor (C1 vs C2)')
plt.xlabel('Componenta Principală 1')
plt.ylabel('Componenta Principală 2')


#graficul corelatiilor dintre matricea x si matricea C
plt.figure(figsize=(10,6))
plt.title('Corelograma modelului')
sb.heatmap(rxcDf,vmin=-1,vmax=1,cmap='RdBu',center=0,annot=True)
plt.xlabel('Componentele principale')
plt.ylabel('Variabilele modelului')

#corelograma comunalitatilor
plt.figure(figsize=(10,6))
plt.title('Corelograma comunalitatilor')
sb.heatmap(comunalitatiDf,cmap='YlGnBu',annot=True)
plt.xlabel('Componente principale')


#scree plot pentru componentele semnificative ( criteriul Kaiser)
plt.figure(figsize=(10,6))
plt.plot(range(1,n+1),alpha,'ro-')
plt.title('Scree Plot (Varianța Componentelor)')
plt.xlabel('Număr Componentă')
plt.ylabel('Varianță (Eigenvalues)')
plt.axhline(y=1,linestyle='--',color='red')

#cercul corelatiilor
plt.figure(figsize=(10,10))
x_coord=r_x_c[:,0]
y_coord=r_x_c[:,1]

plt.title('Cercul corelatiilor')
plt.scatter(x_coord,y_coord,color='blue',edgecolors='black')

for i in range(len(variabile_numerice)):
    plt.arrow(0,0,x_coord[i],y_coord[i],color='blue',alpha=0.8)
    plt.annotate(variabile_numerice[i],(x_coord[i]*1.03,y_coord[i]*1.03))

theta=np.linspace(0,2*np.pi,500)
plt.plot(np.cos(theta),np.sin(theta),color='brown',linestyle='-')

plt.axhline(y=0,color='gray',lw=1,alpha=0.3)
plt.axvline(x=0,color='gray',lw=1,alpha=0.3)

plt.xlabel('Componenta C1')
plt.ylabel('Componenta C2')

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.show()

