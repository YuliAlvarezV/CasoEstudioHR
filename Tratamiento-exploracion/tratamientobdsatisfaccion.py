import pandas as pd ### para manejo de datos
import matplotlib as mpl ## gráficos
import matplotlib.pyplot as plt ### gráficos
import plotly.express as px

encuesta_empleados = 'https://raw.githubusercontent.com/YuliAlvarezV/CasoEstudioHR/main/employee_survey_data.csv'

df_empleados=pd.read_csv(encuesta_empleados)
df_empleados = df_empleados.drop(columns=['Unnamed: 0'])

###### Verificar lectura correcta de los datos

df_empleados.sort_values(by=['EmployeeID'],ascending=1)

##### resumen con información tablas faltantes y tipos de variables y hacer correcciones

df_empleados.info()
df_empleados.isnull().sum()

##### Tratamiento de nulos

df_filled = df_empleados.fillna(0) #### Se reemplaza los valores nulos por ceros, ya que se presume que el empleado se abstuvo de responder

#### Convertir campos a formato fecha 

df_filled["DateSurvey"]=pd.to_datetime(df_filled['DateSurvey'])

#### convertir a categórica

df_filled=df_filled.astype({'EmployeeID': object})
df_filled=df_filled.astype({'EnvironmentSatisfaction': object})
df_filled=df_filled.astype({'JobSatisfaction': object})
df_filled=df_filled.astype({'WorkLifeBalance': object})

df_filled.info()

###################################

#Gráfico de pastel
EnvironmentSatisfaction = df_filled["EnvironmentSatisfaction"].value_counts()

# Crear un diccionario de mapeo de valores originales a nuevos nombres
nuevos_nombres = {0: 'No responde', 1: 'Bajo', 2: 'Mediano', 3: "Alto", 4: "Muy alto"}

# Reemplazar los nombres de las etiquetas utilizando el diccionario de mapeo
EnvironmentSatisfaction.index = EnvironmentSatisfaction.index.map(nuevos_nombres)

# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(EnvironmentSatisfaction, labels= EnvironmentSatisfaction.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Nivel de satisfacción con el ambiente laboral")  # Título del gráfico

plt.show()

###################################

JobSatisfaction = df_filled["JobSatisfaction"].value_counts()

# Crear un diccionario de mapeo de valores originales a nuevos nombres
nuevos_nombres = {0: 'No responde', 1: 'Bajo', 2: 'Mediano', 3: "Alto", 4: "Muy alto"}

# Reemplazar los nombres de las etiquetas utilizando el diccionario de mapeo
JobSatisfaction.index = JobSatisfaction.index.map(nuevos_nombres)

# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(JobSatisfaction, labels= JobSatisfaction.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Nivel de satisfacción con el trabajo")  # Título del gráfico

plt.show()

#####################################

WorkLifeBalance = df_filled["WorkLifeBalance"].value_counts()

# Crear un diccionario de mapeo de valores originales a nuevos nombres
nuevos_nombres = {0: 'No responde', 1: 'Malo', 2: 'Aceptable', 3: "Bueno", 4: "Excelente"}

# Reemplazar los nombres de las etiquetas utilizando el diccionario de mapeo
WorkLifeBalance.index = WorkLifeBalance.index.map(nuevos_nombres)

# Crear un gráfico de pastel
plt.figure(figsize=(8, 8))  # Tamaño del gráfico
plt.pie(WorkLifeBalance, labels= WorkLifeBalance.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
plt.axis('equal')  # Aspecto de círculo
plt.title("Nivel de balance entre el trabajo y su vida")  # Título del gráfico

plt.show()