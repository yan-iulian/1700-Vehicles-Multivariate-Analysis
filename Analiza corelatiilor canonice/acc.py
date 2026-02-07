from utils import *
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sb

masini=pd.read_csv('../SET DE DATE/Specificatii masini 2023.csv',index_col=(0,1))
replaceNan(masini)

#performanta pe traseu vs dimensiuni
variabile_set1=['Power(HP)','Torque(Nm)','Top Speed','Acceleration 0-62 Mph (0-100 kph)']
variabile_set2=['Length','Width','Wheelbase','Unladen Weight']

x_orig=masini[variabile_set1].values
y_orig=masini[variabile_set2].values

x=StandardScaler().fit_transform(x_orig)
y=StandardScaler().fit_transform(y_orig)

n,p=x.shape
_,q=y.shape
nr_comp=min(p,q)

#crearea modelului
model_acc=CCA(n_components=nr_comp)
model_acc.fit(x,y)

#etichete
eticheteZ=['Z'+str(i+1) for i in range(nr_comp)]
eticheteU=['U'+str(i+1) for i in range(nr_comp)]
etichete_radacini=['Radacina'+str(i+1) for i in range(nr_comp)]

#scoruri
z,u=model_acc.transform(x,y)
zDf=toDataFrame(z,masini.index,eticheteZ,'Zscore.csv')
uDf=toDataFrame(u,masini.index,eticheteU,'Uscores.csv')

#corelatiile dintre z si u
r_z_u=np.array([
    np.corrcoef(z[:,i],u[:,i])[0,1]
    for i in range(nr_comp)]
)
r_z_u_patrat=r_z_u**2
p_values=test_bartlett(r_z_u_patrat,n,p,q,nr_comp)
pd.DataFrame({
    'R_z_u':r_z_u,
    'R_z_u patrat':r_z_u_patrat,
    'P_value':p_values
},
index=etichete_radacini).to_csv('R_z_u.csv')

#corealatiile dintre x&z si y&u
r_x_z=np.corrcoef(x,z,rowvar=False)[:nr_comp,nr_comp:]
r_x_zDf=toDataFrame(r_x_z,variabile_set1,eticheteZ,'R_x_z.csv')
r_y_u=np.corrcoef(y,u,rowvar=False)[:nr_comp,nr_comp:]
r_y_uDf=toDataFrame(r_y_u,variabile_set2,eticheteU,'R_y_u.csv')

#variante si redundante
v_x_z=np.mean(r_x_z**2,axis=0)
v_y_u=np.mean(r_y_u**2,axis=0)
redundanta_x=v_x_z*r_z_u_patrat
reduntanta_y=v_y_u*r_z_u_patrat

pd.DataFrame(
    {
        'V_x_z':v_x_z,
        'Redundanta x':redundanta_x,
        'V_y_u':v_y_u,
        'Redundanta y':reduntanta_y
    },index=etichete_radacini
).to_csv('Variante si redundante.csv')

#grafice

#heatmap pentru corelatiile dintre variabilele initiale si componente create

plt.figure(figsize=(10,6))
plt.title('Corelograma variabile set 1 vs Scoruri Z')
sb.heatmap(r_x_zDf, annot=True, cmap='RdBu', vmin=-1, vmax=1)

plt.figure(figsize=(10,6))
plt.title('Corelograma variabile set 2 vs Scoruri U')
sb.heatmap(r_y_uDf, annot=True, cmap='RdBu', vmin=-1, vmax=1)

#scatter plot pentru a arata distributia observatiilor in legatura z1,u1 (cea mai importanta componenta)
plt.figure(figsize=(10,6))
plt.title('Biplot Instanțe: Z1 vs U1')
plt.scatter(z[:, 0], u[:, 0], alpha=0.6, color='green')
plt.xlabel('Z1 (Performanță)')
plt.ylabel('U1 (Dimensiuni)')
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')

#cercul corelatiilor dintre x si z
plt.figure(figsize=(10,10))
plt.title('Cercul corelatiilor')
x_coords=r_x_z[:,0]
y_coords=r_x_z[:,1]
plt.scatter(x_coords,y_coords,color='orange',edgecolors='black')

theta=np.linspace(0,2*np.pi,100)
plt.plot(np.cos(theta),np.sin(theta),color='blue',linestyle='-')

for i in range(len(variabile_set1)):
    plt.arrow(0,0,x_coords[i],y_coords[i],color='orange')
    plt.annotate(variabile_set1[i],(x_coords[i],y_coords[i]))

plt.xlabel('Z1')
plt.ylabel('Z2')
plt.axhline(y=0,alpha=0.3,color='gray')
plt.axvline(x=0,color='gray',alpha=0.3)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

#cercul corelatiilor dintre y si u
plt.figure(figsize=(10,10))
plt.title('Cercul corelatiilor dintre y si u')
k_coords=r_y_u[:,0]
l_coords=r_y_u[:,1]
plt.scatter(k_coords,l_coords,color='violet',edgecolors='green')

theta=np.linspace(0,2*np.pi,100)
plt.plot(np.cos(theta),np.sin(theta),color='red')

for i in range( len(variabile_set2)):
    plt.arrow(0,0,k_coords[i],l_coords[i],alpha=0.8,color='green')
    plt.annotate(variabile_set2[i],(k_coords[i],l_coords[i]))

plt.xlabel('U1')
plt.ylabel('U2')
plt.axhline(y=0,alpha=0.3,color='gray')
plt.axvline(x=0,color='gray',alpha=0.3)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.show()

