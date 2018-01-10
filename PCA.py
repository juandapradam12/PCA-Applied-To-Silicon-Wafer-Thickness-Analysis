import csv 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D

#### LEE ARCHIVO DE DATOS Y GUARDA VARIABLES EN ARRAYS

Datos=np.genfromtxt('siliconwaferthickness.csv', delimiter=',',skip_header=True)

G1,G2,G3,G4,G5,G6,G7,G8,G9=(Datos[:,i] for i in range(9))

#### NORMALIZA LOS DATOS

G=[G1,G2,G3,G4,G5,G6,G7,G8,G9]

g1,g2,g3,g4,g5,g6,g7,g8,g9=([] for i in range(9))
g=[g1,g2,g3,g4,g5,g6,g7,g8,g9]
for i in range(0,G1.shape[0]):
	for gi,Gi in zip(g,G):
		gi.append(Gi[i]-Gi.mean()/Gi.std())

# Matriz de datos normalizados M_datos 184x9

# datos es la matriz de datos normalizados 9x184

datos=np.vstack((g1,g2,g3,g4,g5,g6,g7,g8,g9))

# M_datos es la matriz de datos normalizados 184x9

M_datos=datos.transpose()

#### Grafica de Datos normalizados 


cm = plt.get_cmap('gist_rainbow')
colores=([cm(1.*i/9) for i in range(9)])

for gi, c in zip(g, colores):
	plt.plot(gi, color=c)
plt.title('Oblea de Silicio')
plt.xlabel('Seccion')
plt.ylabel('Espesor')
plt.savefig('ExploracionDatos.pdf')
plt.clf()

#### Calculo Matriz de Covarianza (implementada)

# Coeficientes de la Matriz

def Cov(x,y):
	n=(np.shape(x)[0]+np.shape(y)[0])/2
	suma=0.0
	Cov=0.0
	for i in range(0,n):
		suma+=(x[i]-np.mean(x))*(y[i]-np.mean(y))
	Cov=(1./(n-1.))*suma
	return(Cov)


# Matriz

fila_0,fila_1,fila_2,fila_3,fila_4,fila_5,fila_6,fila_7,fila_8=([] for i in range(9))
filas=[fila_0,fila_1,fila_2,fila_3,fila_4,fila_5,fila_6,fila_7,fila_8]

for j in range(0,9):
	fila_0.append(Cov(M_datos[:,0],M_datos[:,j]))
	fila_1.append(Cov(M_datos[:,1],M_datos[:,j]))
	fila_2.append(Cov(M_datos[:,2],M_datos[:,j]))
	fila_3.append(Cov(M_datos[:,3],M_datos[:,j]))
	fila_4.append(Cov(M_datos[:,4],M_datos[:,j]))
	fila_5.append(Cov(M_datos[:,5],M_datos[:,j]))
	fila_6.append(Cov(M_datos[:,6],M_datos[:,j]))
	fila_7.append(Cov(M_datos[:,7],M_datos[:,j]))
	fila_8.append(Cov(M_datos[:,8],M_datos[:,j]))



# convierto las listas de tipo array

cov_fila_0,cov_fila_1,cov_fila_2,cov_fila_3,cov_fila_4,cov_fila_5,cov_fila_6,cov_fila_7,cov_fila_8=([] for i in range(9))
cov_filas=[cov_fila_0,cov_fila_1,cov_fila_2,cov_fila_3,cov_fila_4,cov_fila_5,cov_fila_6,cov_fila_7,cov_fila_8]

cov_fila_0=np.asarray(fila_0)
cov_fila_1=np.asarray(fila_1)
cov_fila_2=np.asarray(fila_2)				
cov_fila_3=np.asarray(fila_3)
cov_fila_4=np.asarray(fila_4)
cov_fila_5=np.asarray(fila_5)
cov_fila_6=np.asarray(fila_6)
cov_fila_7=np.asarray(fila_7)
cov_fila_8=np.asarray(fila_8)

# Hago la matriz juntando las filas 

M_Cov=np.stack((cov_fila_0,cov_fila_1,cov_fila_2,cov_fila_3,cov_fila_4,cov_fila_5,cov_fila_6,cov_fila_7,cov_fila_8))

#### Imprimir autovalores y autovectores 

val_p, vec_p=linalg.eig(M_Cov) # encuentro los valores y vectores propios n necesariamente en orden

# par ordenado de valores y vectores propios

valyvec_p=[(np.abs(val_p[i]), vec_p[:,i]) for i in range(len(val_p))]

valyvec_p.sort() # los ubica en orden de menor a mayor valor propio 
valyvec_p.reverse() # cambia el orden de mayor a menor valor propio de el par ordenado

print('Valores y Vectores Propios:')
for par in valyvec_p:
    print(par[0], par[1])

PC=np.stack((valyvec_p[0][1], valyvec_p[1][1])).T # matriz de las componentes principales 9x2

# Para describir la variabilidad de los datos elijo dos componentes principales ya que tienen los valores propios con mayor magnitud y estan ubicados en las ultimas dos posiciones de la lista (orden de menor a mayor)

#### Graficar los datos en las dos componentes principales

M_datos_PC=np.dot(M_datos, PC) # cambio de base los datos a la base de componentes principales, M_datos_PC 184x2

plt.plot(M_datos_PC[:,0],M_datos_PC[:,1],'o')
plt.title('Datos respecto a las CPs')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.savefig('PCAdatos.pdf')
plt.clf()

#### Graficar agrupaciones de los datos originales respecto a la base de los componentes principales

cm = plt.get_cmap('gist_rainbow')
colores=([cm(1.*i/9) for i in range(9)])

for i, c in zip(range(9), colores):
	plt.scatter(PC[i,0], PC[i,1], color=c)
plt.title('Agrupacion de datos originales respecto a CP')
plt.legend(["g1","g2","g3","g4","g5","g6","g7","g8","g9"])
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.savefig('PCAvariables.pdf')
plt.clf()

#### Imprime 

# G1 y G3 son las variables que se encuentran mas cerca entre si, se podria pensar que G4 tambien. Pero G1 y G3 estan mas cerca respecto a al eje de la componente principal 1, que tiene la mayor cantidad de informacion del sistema

print("Las variables que estan correlacionadas son G1 y G3 donde G1 y G3 corresponde a los grupos de variables encontrados.")

