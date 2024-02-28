#### Cargar paquetes siempre al inicio

import pandas as pd   ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
from pandas.plotting import scatter_matrix  ## para matriz de correlaciones
from sklearn import tree ###para ajustar arboles de decisión
from sklearn.tree import export_text ## para exportar reglas del árbol
#import a_funciones as funciones  ###archivo de funciones propias

### Cargar tablas de datos desde github ###
   
general='https://raw.githubusercontent.com/YuliAlvarezV/CasoEstudioHR/main/Bases%20de%20datos/general_data.csv'

df_general=pd.read_csv(general)
df_general.columns

###### Verificar lectura correcta de los datos

df_general.sort_values(by=['EmployeeID'],ascending=1).head(100)

##### resumen con información tablas faltantes y tipos de variables y hacer correcciones

df_general.info(verbose=True)
#### Tratamiento de nulos, para la variable TotalWorkingYears vemos que hay 18 nulos por ello decidimos utilizar el metodo de llenarlos con la media ya que es informacion que puede ayudarnos mas adelante
## de la misma forma trataremos los nulos en NumCompaniesWorked

df_general['TotalWorkingYears'] = df_general['TotalWorkingYears'].fillna(df_general['TotalWorkingYears'].mean())
df_general['NumCompaniesWorked'] = df_general['NumCompaniesWorked'].fillna(df_general['NumCompaniesWorked'].mean())
###Confirmamos los cambios
df_general.info(verbose=True)

#### Convertir campos a formato fecha 
df_general["InfoDate"]=pd.to_datetime(df_general['InfoDate'])
#### convertir a categórica
df_general=df_general.astype({'EmployeeID': object,"EmployeeID": object})

###Eliminar columnas que no se utilicen
df_general=df_general.drop(['Unnamed: 0','StandardHours', 'Over18', 'EmployeeCount'], axis=1) # Unnamed no aporta datos relevantes, #EmployeeCount siempre sera 1,#Standard Hours siempre sera 8 y no b rinda mayor diferenciación para el analisis y Over18 tiene el mismo valor para todos ya que todos son mayores de edad

####explorar variables numéricas con histograma
fig=df_general.hist(bins=50, figsize=(40,30),grid=False,ec='black')
plt.show()

####explorar variables categoricas
BT = df_general["BusinessTravel"].value_counts()

# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(BT, labels= BT.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Viajes")  # Título del gráfico
plt.show()

Department=df_general['Department'].value_counts()
# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(Department, labels= Department.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Departamento donde trabaja")  # Título del gráfico

plt.show()

EducationField=df_general['EducationField'].value_counts()
# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(EducationField, labels= EducationField.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Estudios")  # Título del gráfico

plt.show()

Gender=df_general['Gender'].value_counts()
# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(Gender, labels= Gender.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Género")  # Título del gráfico

plt.show()

JobRole=df_general['JobRole'].value_counts()
# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(JobRole, labels= JobRole.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Posición del empleado")  # Título del gráfico

plt.show()

MaritalStatus=df_general['MaritalStatus'].value_counts()
# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(MaritalStatus, labels= MaritalStatus.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Estado Civil")  # Título del gráfico

plt.show()