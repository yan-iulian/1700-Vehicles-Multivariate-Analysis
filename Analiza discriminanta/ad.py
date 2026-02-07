import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

#x_train,y_train,x_test,y_test=train_test_split(x,y,0.3,random_state=1,stratify=y)

masini=pd.read_csv('../SET DE DATE/masini_proiect_final_partitionat.csv',index_col=(0,1,2))

masini_train=masini.loc[masini['Partition']=='Antrenament']
masini_test=masini.loc[masini['Partition']=='Test']

variabile_independente=masini.columns[2:15].tolist()
variabila_dependenta='Segment'

x_train=masini_train[variabile_independente].values
y_train=masini_train[variabila_dependenta].values

x_test=masini_test[variabile_independente].values
observatii=masini_test.index

#functie de calcul acuratete si matrice de confuzie
def calculeaza_acuratate(y_true,y_pred,grupe):
    matrice_confuzie=confusion_matrix(y_true,y_pred,labels=grupe)
    ag=np.round(np.diagonal(matrice_confuzie).sum()*100/np.sum(matrice_confuzie),3)
    a_within_g=np.round(np.diagonal(matrice_confuzie)*100/np.sum(matrice_confuzie,axis=1),3)
    am=np.mean(a_within_g)
    return matrice_confuzie,ag,a_within_g,am

#crearea modelului
model_lda=LinearDiscriminantAnalysis()
model_lda.fit(x_train,y_train)
grupe=model_lda.classes_

#testarea
y_train_prezis=model_lda.predict(x_train)
y_test_prezis=model_lda.predict(x_test)

matrice_confuzie,acuratete_globala, acuratate_in_interiorul_grupelor,acuratete_medie=calculeaza_acuratate(y_train,y_train_prezis,grupe)

print('Matrice confuzie:\n',matrice_confuzie)
print('Acuratetea globala: ',acuratete_globala)
print('Acurataetea intre grupuri: ',acuratate_in_interiorul_grupelor)
print('Acuratetea medie: ',acuratete_medie)

pd.DataFrame({
    variabila_dependenta:y_train,
    'Predictia modelului':y_train_prezis
},index=masini_train.index).to_csv('Predictii de train.csv')

pd.DataFrame({
    'Predictia modelului':y_test_prezis
},index=masini_test.index).to_csv('Predictii de test.csv')

z_train=model_lda.transform(x_train)
means=model_lda.means_
z_means=model_lda.transform(means)

#grafice

#graficul primelor doua componente noi gasite la reducerea dimensionalitatii
plt.figure(figsize=(10,6))
plt.title('Scatter plot cu primele doua componente')
plt.xlabel('Componenta principala C_ADL1')
plt.ylabel('Componenta principala C_ADL2')
ax=sb.scatterplot(x=z_train[:,0],y=z_train[:,1],hue=y_train_prezis,palette='viridis')
sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

#graficul distributiei grupelor pe prima axa C_ADL1
plt.figure(figsize=(10,6))
bx=plt.gca()
plt.title('Graficul distributiei grupelor pe prima axa C_ADL1')
grupe_unice=np.unique(y_train)
for g in grupe_unice:
    date_axa1 = z_train[y_train == g]
    valori = date_axa1[:, 0]
    sb.kdeplot(valori,fill=True,label=str(g))
plt.legend()
sb.move_legend(bx, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Axa Discriminantă Z1')
plt.ylabel('Densitate')

#graficul matricei de confuzie
plt.figure(figsize=(10,8))
plt.title('Graficul matricei de confuzie')
sb.heatmap(matrice_confuzie,
           cmap='Blues',
           annot=True,
           fmt='d',
           xticklabels=grupe_unice,
           yticklabels=grupe_unice,
           linewidths=.5,
           cbar_kws={'label': 'Număr observații'})
plt.xticks(rotation=45, ha='right')
plt.xlabel('Etichete Prezise')
plt.ylabel('Etichete Reale')

#reprezentarea vizuala a centroizilor si a scorurilor obtinute din reducerea dimensionalitatii
n_axe=min(len(variabile_independente),len(grupe)-1)
print(f"Numarul de axe este {n_axe}")   #13

plt.figure(figsize=(10,10))
cx=plt.gca()
plt.title('Scatter plot pentru scoruri si centroizi')
sb.scatterplot(x=z_train[:,0],y=z_train[:,1],hue=y_train,hue_order=grupe)
sb.scatterplot(x=z_means[:,0],y=z_means[:,1],hue=grupe,marker='s',s=255)
sb.move_legend(cx, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('LD1')
plt.ylabel('LD2')

plt.show()
print()
