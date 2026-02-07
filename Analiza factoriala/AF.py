import numpy as np
from factor_analyzer import FactorAnalyzer,calculate_kmo,calculate_bartlett_sphericity
import pandas as pd
import sys
from utils import *
import matplotlib.pyplot as plt
import seaborn as sb

masini=pd.read_csv('../SET DE DATE/Specificatii masini 2023.csv',index_col=(0,1))
replaceNan(masini)
variabile=masini.columns[2:].tolist()
x=masini[variabile].values

#testele de compatibilitate
chi2,p_value=calculate_bartlett_sphericity(x)
print(f'Chi patrat: {chi2}\nP-value: {p_value}')
if p_value<0.05:
    print('H(0): variabilele sunt independente (RESPINS)')
    print('H(1): variabilele nu sunt independente (ACCEPT)')
else:
    print('H(0): variabilele sunt independente (ACCEPT)')
    print('H(1): variabilele nu sunt independente (RESPINS)')
    print('NU se poate analiza factoriala')
    sys.exit(-10)

kmo_all,kmo_overall=calculate_kmo(x)
print(f'\nValoarea testului KMO: {kmo_overall}')
if kmo_overall>0.6:
    print(f'Si testul kmo a trecut ({np.round(kmo_overall,3)}>0.6). Se poate aplica analiza factoriala')
else:
    print(f'Valoarea testului kmo este prea mica({np.round(kmo_overall,3)}<0.6). Nu se poate aplica analiza factoriala')
    sys.exit(-1)

#gasirea numarului de factori latenti
fa_n=FactorAnalyzer(rotation=None)
fa_n.fit(x)

valori_proprii,_=fa_n.get_eigenvalues()

conditie=np.where(valori_proprii>1)
n_factori=len(conditie[0])
print(f'In setul de date exista {n_factori} factori latenti')

#initializarea modelului propriu zis
fa=FactorAnalyzer(n_factors=n_factori,rotation='varimax')
fa.fit(x)
eticheteFactori=['F'+str(i+1) for i in range(n_factori)]

#scoruri
scoruri=fa.transform(x)
scoruriDf=toDataFrame(scoruri,masini.index,eticheteFactori,'Scoruri.csv')

#comunalitati si varianta specifica
comunalitati=fa.get_communalities()
varianta_specifica=1-comunalitati
comunalitatiDf=toDataFrame(comunalitati,variabile,['Comunalitati'],'Comunalitati.csv')
comunalitatiDf['Varianta specifica']=varianta_specifica
comunalitatiDf.to_csv('Comunalitati si varianta specifica.csv')

#varianta
varianta_individuala,varianta_prop,varianta_cumulata=fa.get_factor_variance()
pd.DataFrame({
    'Varianta':varianta_individuala,
    'Varianta proportionala':varianta_prop,
    'Varianta cumulata':varianta_cumulata
},index=eticheteFactori).to_csv('Varianta.csv')

#loadings (corelatii factoriale)
loadings=fa.loadings_
loadingsDf=toDataFrame(loadings,variabile,eticheteFactori,'Loadings.csv')


#grafice

#scree plot al valorilor proprii
plt.figure(figsize=(10,6))
plt.title('Scree plot al valorilor proprii')
plt.plot(range(1,len(valori_proprii)+1),valori_proprii,'ro-')
plt.axhline(y=1,color='blue',linestyle=':')
plt.xlabel('Numar factor')
plt.ylabel('Valoare proprie')

#heatmap corelatia dintre factori si variabilele principale
plt.figure(figsize=(10,6))
plt.title('Heatmap loadings')
plt.xlabel('Factori')
plt.ylabel('Variabilele modelului')
sb.heatmap(loadingsDf,vmin=-1,vmax=1,cmap='RdBu',annot=True)


#scatter pentru scoruri
plt.figure(figsize=(10,6))
plt.title('Scatter plot factorii latenti')
plt.scatter(scoruriDf['F1'],scoruriDf['F2'],alpha=0.5,color='red')
plt.xlabel('Factorul latent F1')
plt.ylabel('Factorul latent F2')
plt.axhline(y=0,color='purple')
plt.axvline(x=0,color='purple')


#heat map cu varianta specifica
plt.figure(figsize=(10,6))
plt.title('Heatmap cu varianta specifica')
sb.heatmap(comunalitatiDf,cmap='YlGnBu',annot=True)
plt.ylabel('Variabilele modelului')

#cercul corelatiilor
plt.figure(figsize=(8,8))
x_coords=loadings[:,0]
y_coords=loadings[:,1]

plt.title('Cercul corelatiilor pentru factorii F1 si F2')
plt.scatter(x_coords,y_coords,color='green',edgecolors='black')

theta=np.linspace(0,2*np.pi,100)
plt.plot(np.cos(theta),np.sin(theta),color='black',linestyle='-')

for i in range(len(variabile)):
    plt.arrow(0,0,x_coords[i],y_coords[i],alpha=0.8,color='red')
    plt.annotate(variabile[i],(x_coords[i]*1.03,y_coords[i]*1.09))

plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.axhline(y=0,color='gray',alpha=0.3)
plt.axvline(x=0,color='gray',alpha=0.3)

plt.xlabel('F1')
plt.ylabel('F2')

plt.show()

