import CodigoVS as basegen
import tratamientobdsatisfaccion as basesat
import manager as basemg
import retirados as basert
import sys ## saber ruta de la que carga paquetes
import pandas as pd
import plotly as pt
import seaborn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Users\\yulia\\Desktop\\EntregaHRAnalitica\\CasoEstudioHR\\Tratamiento-exploracion') ## este comanda agrega una ruta

general = basegen.df_general
satisfaccion = basesat.df_filled
manager = basemg.manager
retirados = basert.df_retirados

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

# Inicializa el LabelEncoder
encoder = LabelEncoder()

# Ajusta el LabelEncoder a la columna 'bussiness' y transforma las categorías en números
result['bussiness_encoded'] = encoder.fit_transform(result['BusinessTravel'])
result[['BusinessTravel','bussiness_encoded']]
result['bussiness_encoded'].unique()

result['education_encoded'] = encoder.fit_transform(result['EducationField'])
result[['EducationField','education_encoded']]
result['EducationField'].unique()
result['education_encoded'].unique()

result['department_encoded'] = encoder.fit_transform(result['Department'])
result[['Department','department_encoded']]
result['Department'].unique()
result['department_encoded'].unique()

result['gender_encoded'] = encoder.fit_transform(result['Gender'])
result[['Gender','gender_encoded']]
result['Gender'].unique()
result['gender_encoded'].unique()

result['jobrole_encoded'] = encoder.fit_transform(result['JobRole'])
result[['JobRole','jobrole_encoded']]
result['JobRole'].unique()
result['jobrole_encoded'].unique()

result['maritalstatus_encoded'] = encoder.fit_transform(result['MaritalStatus'])
result[['MaritalStatus','maritalstatus_encoded']]
result['MaritalStatus'].unique()
result['maritalstatus_encoded'].unique()

result.info()

result['Attrition'] = result['Attrition'].fillna('No aplica')
result['retirementType'] = result['Attrition'].fillna('No aplica')
result['resignationReason'] = result['Attrition'].fillna('No aplica')

result['attrition_encoded'] = encoder.fit_transform(result['Attrition'])
result[['Attrition','attrition_encoded']]
result['Attrition'].unique()
result['attrition_encoded'].unique()

result['retirementtype_encoded'] = encoder.fit_transform(result['retirementType'])
result[['retirementType','retirementtype_encoded']]
result['retirementType'].unique()
result['retirementtype_encoded'].unique()

result['resignationreason_encoded'] = encoder.fit_transform(result['resignationReason'])
result[['resignationReason','resignationreason_encoded']]
result['resignationReason'].unique()
result['resignationreason_encoded'].unique()

######## Eliminacion de variables

Eliminar = ['BusinessTravel', 'EducationField', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'Attrition', 'retirementType', 'resignationReason']
result = result.drop(Eliminar, axis=1)

#####Separacion variable respuesta

Y = result['attrition_encoded']
X = result.drop(['attrition_encoded'],axis= 1 )

####Escalado

#Cambiamos las variables a int
X['JobInvolvement'] = X['JobInvolvement'].astype(int)
X['PerformanceRating'] = X['PerformanceRating'].astype(int)
X['EnvironmentSatisfaction'] = X['EnvironmentSatisfaction'].astype(int)
X['JobSatisfaction'] = X['JobSatisfaction'].astype(int)
X['WorkLifeBalance'] = X['WorkLifeBalance'].astype(int)

#Seleccionamos las columnas que se escalan 
columnas_a_escalar = ["Age","DistanceFromHome", "Education", "JobLevel","MonthlyIncome", "NumCompaniesWorked", "PercentSalaryHike",
                      "StandardHours", "StockOptionLevel", "TotalWorkingYears","TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion",
                      "YearsWithCurrManager","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction",
                      "bussiness_encoded","education_encoded","department_encoded","gender_encoded","jobrole_encoded","maritalstatus_encoded",
                      "retirementtype_encoded","resignationreason_encoded"]

# Inicializa el escalador
scaler = StandardScaler()

X[columnas_a_escalar] = scaler.fit_transform(X[columnas_a_escalar])

# Imprime el DataFrame resultante
print(X)
X_Scaled=X[columnas_a_escalar]
X_Scaled

#Se dividen los datos en conjuntos de entrenamiento y validación. El 20% de los datos se utilizará para la validación.
X_train, X_valid, y_train, y_valid = train_test_split(X_Scaled,Y,test_size = 0.2 ,stratify=Y, random_state= 1 )

#Creamos y entrenamos el clasificador RandoForest en en los conjuntos
classifier = RandomForestClassifier() 
classifier.fit(X_train,y_train)