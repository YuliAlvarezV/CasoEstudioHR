import CodigoVS as basegen
import tratamientobdsatisfaccion as basesat
import manager as basemg
import retirados as basert
import sys ## saber ruta de la que carga paquetes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

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
result


#### Tratamiento de nulos en varible respuesta y variables de la tabla retirados

result['Attrition'] = result['Attrition'].fillna('No')
result['retirementType'] = result['retirementType'].fillna('No aplica')
result['resignationReason'] = result['resignationReason'].fillna('No aplica')
result['retirementDate'] = result['retirementDate'].fillna('No aplica')


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

result=result.astype({'EnvironmentSatisfaction': float})
result=result.astype({'JobSatisfaction': float})
result=result.astype({'WorkLifeBalance': float})
result=result.astype({'JobInvolvement': float})
result=result.astype({'PerformanceRating': float})

### Dummies

result = pd.get_dummies(result, dummy_na = True)
result


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
columnas_a_escalar = ["Age","DistanceFromHome", "Education", "JobLevel","MonthlyIncome", "PercentSalaryHike",
                      "StandardHours", "StockOptionLevel","TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion",
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

# Realizar predicción 
preds = classifier.predict(X_valid) 

#Se valida el desempeño
accuracy_score(preds,y_valid)

# Seleccion automatica de caracterirsticas usando featurewiz  
target = 'attrition_encoded'
features, train = featurewiz(result, target, corr_limit= 0.7 , verbose= 2 , sep= "," , header= 0 ,test_data= "" , feature_engg= "" , category_encoders= "" )


print(features)