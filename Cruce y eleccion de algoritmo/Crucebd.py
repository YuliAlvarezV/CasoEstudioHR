import CodigoVS as basegen
import tratamientobdsatisfaccion as basesat
import manager as basemg
import retirados as basert
import funciones as func
import sys ## saber ruta de la que carga paquetes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
import seaborn as sns
import matplotlib.pyplot as plt


###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\Users\GOMEZ\Documents\2024-1\ANALITICA\CasoEstudioHR\Tratamiento-exploracion') ## este comanda agrega una ruta


general = basegen.df_general
satisfaccion = basesat.df_filled
manager = basemg.manager
retirados = basert.df_retirados
general.info()

##### Separación de datos por año

general_2015 = general[general['InfoDate'].dt.year == 2015]
general_2016 = general[general['InfoDate'].dt.year == 2016]

satisfaccion_2015 = satisfaccion[satisfaccion['DateSurvey'].dt.year == 2015]
satisfaccion_2016 = satisfaccion[satisfaccion['DateSurvey'].dt.year == 2016]

manager_2015 = manager[manager['SurveyDate'].dt.year == 2015]
manager_2016 = manager[manager['SurveyDate'].dt.year == 2016]

retirados_2015 = retirados[retirados['retirementDate'].dt.year == 2015]
retirados_2016 = retirados[retirados['retirementDate'].dt.year == 2016]

#### Cruce de bds 2015

df_merge2015 = pd.merge(general_2015, manager_2015, left_on='EmployeeID', right_on='EmployeeID')
df_merge2015 = pd.merge(df_merge2015, satisfaccion_2015, left_on='EmployeeID', right_on='EmployeeID')
df_merge2015 = df_merge2015.merge(retirados_2015, on='EmployeeID', how='left')

df_merge2015.shape
df_merge2015.info()


##### Cruce de bds 2016

df_merge2016 = pd.merge(general_2016, manager_2016, left_on='EmployeeID', right_on='EmployeeID')
df_merge2016 = pd.merge(df_merge2016, satisfaccion_2016, left_on='EmployeeID', right_on='EmployeeID')
df_merge2016 = df_merge2016.merge(retirados_2016, on='EmployeeID', how='left')

df_merge2016.shape
df_merge2016.info()

############# Cruce base de BDs completas

result = pd.concat([df_merge2015, df_merge2016])
result


#### Tratamiento de nulos en varible respuesta y variables de la tabla retirados

result['Attrition'] = result['Attrition'].fillna('No')
result['retirementType'] = result['retirementType'].fillna('No aplica')
result['resignationReason'] = result['resignationReason'].fillna('No aplica')
result['retirementDate'] = result['retirementDate'].fillna('No aplica')

###ANALISIS BIVARIADO DE TABLA GENERAL Y VARIABLE RESPUESTA#######
#Variable BusinessTravel vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='BusinessTravel', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por Viajes')

# Mostrar el gráfico
plt.show()

#Variable Department vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='Department', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por Departamento')

# Mostrar el gráfico
plt.show()

#Variable EducationField vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='EducationField', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por Campo de estudio')

# Mostrar el gráfico
plt.show()

#Variable Education vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='Education', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por nivel de educación')

# Mostrar el gráfico
plt.show()


#Variable Gender vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='Gender', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por genero')

# Mostrar el gráfico
plt.show()

#Variable JobLevel vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='JobLevel', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por nivel de trabajo')

# Mostrar el gráfico
plt.show()

#Variable JobRole vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='JobRole', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por Rol')

# Mostrar el gráfico
plt.show()

#Variable MaritalStatus vs Atrittion
# Crear el gráfico de barras apiladas utilizando Seaborn
sns.countplot(x='MaritalStatus', hue='Attrition', data=result)

# Añadir título al gráfico
plt.title('Distribución de la Variable Respuesta por Estado civil')

# Mostrar el gráfico
plt.show()

#Variable Age vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "Age"
sns.boxplot(x='Attrition', y='Age', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Age')

# Mostrar el gráfico
plt.show()

#Variable DistanceFromHome vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "Age"
sns.boxplot(x='Attrition', y='DistanceFromHome', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con distancia de casa')

# Mostrar el gráfico
plt.show()

#Variable MonthlyIncome vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "MonthlyIncome"
sns.boxplot(x='Attrition', y='MonthlyIncome', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Ingreso mensual en Rupias')

# Mostrar el gráfico
plt.show()

#Variable NumCompaniesWorked vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "NumCompaniesWorked"
sns.boxplot(x='Attrition', y='NumCompaniesWorked', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número total de empresas para las que ha trabajado el empleado')

# Mostrar el gráfico
plt.show()

#Variable PercentSalaryHike vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "PercentSalaryHike"
sns.boxplot(x='Attrition', y='PercentSalaryHike', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Aumento salarial porcentual del año pasado')

# Mostrar el gráfico
plt.show()

#Variable StockOptionLevel vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "StockOptionLevel"
sns.boxplot(x='Attrition', y='StockOptionLevel', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Nivel de opción de compra del empleado')

# Mostrar el gráfico
plt.show()

#Variable TotalWorkingYears vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "TotalWorkingYears"
sns.boxplot(x='Attrition', y='TotalWorkingYears', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número total de años que el empleado ha trabajado hasta el momento')

# Mostrar el gráfico
plt.show()

#Variable TrainingTimesLastYear vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "TrainingTimesLastYear"
sns.boxplot(x='Attrition', y='TrainingTimesLastYear', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número de veces que se realizó capacitación para este empleado el año pasado')

# Mostrar el gráfico
plt.show()

#Variable YearsAtCompany vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "YearsAtCompany"
sns.boxplot(x='Attrition', y='YearsAtCompany', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número total de años que el empleado ha permanecido en la empresa')

# Mostrar el gráfico
plt.show()

#Variable YearsSinceLastPromotion vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "YearsSinceLastPromotion"
sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número de años desde la última promoción')

# Mostrar el gráfico
plt.show()

#Variable YearsWithCurrManager vs Atrittion
# Crear el gráfico de caja para comparar "Attrition" con "YearsWithCurrManager"
sns.boxplot(x='Attrition', y='YearsWithCurrManager', data=result)

# Añadir título al gráfico
plt.title('Comparación de Attrition con Número de años bajo el mando actual')

# Mostrar el gráfico
plt.show()

# Inicializa el LabelEncoder
encoder = LabelEncoder()

######### Encoder para la variable respuesta

result['attrition_encoded'] = encoder.fit_transform(result['Attrition'])
result[['Attrition','attrition_encoded']]
result['Attrition'].unique()
result['attrition_encoded'].unique()

######## Eliminacion de variables

Eliminar = ['Attrition', 'DateSurvey', 'retirementDate', 'InfoDate', 'SurveyDate', 'EmployeeID']
#### dummies 'BusinessTravel', 'EducationField', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'retirementType', 'resignationReason
result = result.drop(Eliminar, axis=1)
result.info()

result=result.astype({'EnvironmentSatisfaction': int})
result=result.astype({'JobSatisfaction': int})
result=result.astype({'WorkLifeBalance': int})
result=result.astype({'JobInvolvement': int})
result=result.astype({'PerformanceRating': int})

### Dummies

result = pd.get_dummies(result, dummy_na = True)
result


#####Separacion variable respuesta

Y = result['attrition_encoded']
X = result.drop(['attrition_encoded'],axis= 1 )
X.info()

####Escalado

#Seleccionamos las columnas que se escalan 
columnas_a_escalar = ["Age","DistanceFromHome", "MonthlyIncome", 'NumCompaniesWorked', "PercentSalaryHike"
                    , "StockOptionLevel", 'TotalWorkingYears',"TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion",
                    "YearsWithCurrManager","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction", 'WorkLifeBalance']

# Inicializa el escalador
##scaler = StandardScaler()

##X[columnas_a_escalar] = scaler.fit_transform(X[columnas_a_escalar])

# escaralar variables

func.escalado(X, columnas_a_escalar)

# Imprime el DataFrame resultante
print(X)

#Se dividen los datos en conjuntos de entrenamiento y validación. El 20% de los datos se utilizará para la validación.
X_train, X_valid, y_train, y_valid = train_test_split(X,Y,test_size = 0.2 ,stratify=Y, random_state= 1 )

