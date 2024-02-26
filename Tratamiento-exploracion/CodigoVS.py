#### Cargar paquetes siempre al inicio

import pandas as pd   ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
#import a_funciones as funciones  ###archivo de funciones propias




####################################################################################################################
########################  1. Comprender y limpiar datos ##################################################################
####################################################################################################################
########   Verificar lectura correcta de los datos
########   Verificar Datos faltantes (eliminar variables si es necesario) (la imputación está la parte de modelado)
########   Tipos de variables (categoricas/numéricas/fechas)
########   Niveles en categorícas 
########   Observaciones por categoría
########   Datos atípicos en numéricas


### Cargar tablas de datos desde github ###
   
general='https://raw.githubusercontent.com/YuliAlvarezV/CasoEstudioHR/main/Bases%20de%20datos/general_data.csv'

df_general=pd.read_csv(general)
df_general.columns

###### Verificar lectura correcta de los datos

df_general.sort_values(by=['EmployeeID'],ascending=1).head(100)

##### resumen con información tablas faltantes y tipos de variables y hacer correcciones

df_general.info(verbose=True)

#### Convertir campos a formato fecha 
df_general["InfoDate"]=pd.to_datetime(df_general['InfoDate'])
#### convertir a categórica
df_general=df_general.astype({'EmployeeID': object,"EmployeeID": object})

###Eliminar columnas que no se utilicen
df_general=df_general.drop(['Unnamed: 0', 'Over18', 'EmployeeCount'], axis=1) # Unnamed no aporta datos relevantes, #EmployeeCount siempre sera 1 y Over18 tiene el mismo valor para todos ya que todos son mayores de edad

####explorar variables numéricas con histograma
fig=df_general.hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show()
