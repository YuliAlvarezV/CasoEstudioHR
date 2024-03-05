import funciones as func
import sys ## saber ruta de la que carga paquetes
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import Crucebd as bd

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\Cruce-eleccion-algoritmo') ## este comanda agrega una ruta

# Inicializa el LabelEncoder
encoder = LabelEncoder()

### Se importa datos de 2015 y 2016

df_merge2015 = bd.df_merge2015
df_merge2016 = bd.df_merge2016

#### Tratamiento de nulos en varible respuesta y variables de la tabla retirados

df_merge2015['Attrition'] = df_merge2015['Attrition'].fillna('No')
df_merge2016['Attrition'] = df_merge2016['Attrition'].fillna('No')
df_merge2015['retirementType'] = df_merge2015['retirementType'].fillna('No aplica')
df_merge2016['retirementType'] = df_merge2016['retirementType'].fillna('No aplica')
df_merge2015['resignationReason'] = df_merge2015['resignationReason'].fillna('No aplica')
df_merge2016['resignationReason'] = df_merge2016['resignationReason'].fillna('No aplica')
df_merge2015['retirementDate'] = df_merge2015['retirementDate'].fillna('No aplica')
df_merge2016['retirementDate'] = df_merge2016['retirementDate'].fillna('No aplica')

######### Encoder para la variable respuesta

df_merge2015['attrition_encoded'] = encoder.fit_transform(df_merge2015['Attrition'])
df_merge2016['attrition_encoded'] = encoder.fit_transform(df_merge2016['Attrition'])
df_merge2015[['Attrition','attrition_encoded']]
df_merge2016[['Attrition','attrition_encoded']]
df_merge2015['Attrition'].unique()
df_merge2016['Attrition'].unique()
df_merge2015['attrition_encoded'].unique()
df_merge2016['attrition_encoded'].unique()

######## Eliminacion de variables

Eliminar = ['Attrition', 'DateSurvey', 'retirementDate', 'InfoDate', 'SurveyDate', 'EmployeeID']

df_merge2015 = df_merge2015.drop(Eliminar, axis=1)
df_merge2016 = df_merge2016.drop(Eliminar, axis=1)
df_merge2015.info()
df_merge2016.info()

df_merge2015=df_merge2015.astype({'EnvironmentSatisfaction': int})
df_merge2016=df_merge2016.astype({'EnvironmentSatisfaction': int})
df_merge2015=df_merge2015.astype({'JobSatisfaction': int})
df_merge2016=df_merge2016.astype({'JobSatisfaction': int})
df_merge2015=df_merge2015.astype({'WorkLifeBalance': int})
df_merge2016=df_merge2016.astype({'WorkLifeBalance': int})
df_merge2015=df_merge2015.astype({'JobInvolvement': int})
df_merge2016=df_merge2016.astype({'JobInvolvement': int})
df_merge2015=df_merge2015.astype({'PerformanceRating': int})
df_merge2016=df_merge2016.astype({'PerformanceRating': int})
df_merge2015=df_merge2015.astype({'Education': int})
df_merge2016=df_merge2016.astype({'Education': int})
df_merge2015=df_merge2015.astype({'JobLevel': int})
df_merge2016=df_merge2016.astype({'JobLevel': int})


### Dummies

df_merge2015 = pd.get_dummies(df_merge2015, dummy_na = True)
df_merge2016 = pd.get_dummies(df_merge2016, dummy_na = True)
df_merge2015
df_merge2016.info()


#####Separacion variable respuesta

Y = df_merge2016['attrition_encoded']
X_train = df_merge2015.drop(['attrition_encoded'],axis= 1 )
X_test = df_merge2016.drop(['attrition_encoded'],axis= 1 )
X_train.info()
X_test.info()

####Escalado

#Seleccionamos las columnas que se escalan 
columnas_a_escalar = ["Age","DistanceFromHome", "MonthlyIncome", 'NumCompaniesWorked', "PercentSalaryHike", "StockOptionLevel", 'TotalWorkingYears',"TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion", "YearsWithCurrManager","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction", 'WorkLifeBalance', 'Education', 'JobLevel']

# Inicializa el escalador
scaler = StandardScaler()

X_train[columnas_a_escalar] = scaler.fit_transform(X_train[columnas_a_escalar])
X_test[columnas_a_escalar] = scaler.fit_transform(X_test[columnas_a_escalar])

# escaralar variables

# func.escalado(X_train, columnas_a_escalar)
# func.escalado(X_test, columnas_a_escalar)

# Imprime el DataFrame resultante
print(X_train)
print(X_test)

### Elección de variables

m_lreg = LogisticRegression(max_iter=200)
m_rtree= DecisionTreeClassifier()
m_rf= RandomForestClassifier()
m_gbc = GradientBoostingClassifier()

modelos=list([m_lreg,m_rtree, m_rf, m_gbc])

var_names=func.sel_variables(modelos,X_train,Y,threshold="2.5*mean")
var_names.shape

X2=X_train[var_names] ### matriz con variables seleccionadas
X2.info()
X_train.info()