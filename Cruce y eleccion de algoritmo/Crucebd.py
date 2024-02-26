import CodigoVS as basegen
import tratamientobdsatisfaccion as basesat
import manager as basemg
import retirados as basert
import sys ## saber ruta de la que carga paquetes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\Tratamiento-exploracion') ## este comanda agrega una ruta

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

X = result.drop(['attrition_encoded'],axis= 1 )

####Escalado

# Inicializa el escalador
scaler = StandardScaler()

# Escala las variables del DataFrame
df_scaled = scaler.fit_transform(X)

# Crea un nuevo DataFrame con las variables escaladas
df_scaled = pd.DataFrame(df_scaled, columns=result.columns)

# Imprime el DataFrame resultante
print(df_scaled)