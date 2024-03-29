# -*- coding: utf-8 -*-
"""Retirados.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PmbzNFkmtl8UE4UHn8mvQiIPsgNMwk5K
"""

#####paquete básicos ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""##Cargar base de datos y eliminar columnas innecesarias"""

#from google.colab import drive
#drive.mount('/content/drive')

# se carga la base de datos
df_retirados = "https://raw.githubusercontent.com/YuliAlvarezV/CasoEstudioHR/main/Bases%20de%20datos/retirement_info.csv"
df_retirados = pd.read_csv(df_retirados)
df_retirados


###Se eliminan columnas innecesarias
df_retirados = df_retirados.drop(columns=['Unnamed: 0.1'])
df_retirados = df_retirados.drop(columns=['Unnamed: 0'])
df_retirados

###Verificar lectura correcta de los datos
df_retirados.sort_values(by=['EmployeeID'],ascending=1).head(100)

"""##Analizar la información de la base de datos, nulos y tipo de variables"""

df_retirados.info()

##Se cambia el tipo de variable retirementDate a fecha
df_retirados['retirementDate']=pd.to_datetime(df_retirados['retirementDate'])

##Se cambia el tipo de variable EmployeeID a categorica
df_retirados = df_retirados.astype({'EmployeeID': object})

##Se cuentan los nulos
df_retirados.isnull().sum()

#### Se elimina los registros con valor fired(despedidos) de la variable retirementType debido a que no aporta ningun valor
#### para nuestro estudio, ya que se busca controlar las renuncias y no los despidos

df_retirados = df_retirados.drop(df_retirados[df_retirados['retirementType'] == 'Fired'].index)

##Se quita la variable resignationReason y  debido a que esta varible solo nos dice la razon de retiros, y llenarlo con otra variable los nulos los que haria es llenar los registros que no se retiraron, por lo que no nos interesa tener estas variables en el modelo
df_retirados = df_retirados.drop(['resignationReason'], axis=1)
df_retirados = df_retirados.drop(['retirementType'], axis=1)

df_retirados.isnull().sum()

##Contar los valores unicos de retirementDate
df_retirados['retirementDate'].value_counts()