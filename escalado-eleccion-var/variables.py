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
import matplotlib.pyplot as plt ### gráficos
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib  ### para guardar modelos
import Crucebd as bd

###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\Cruce-eleccion-algoritmo') ## este comanda agrega una ruta

### Se importa datos de 2015 y 2016

df_merge2015 = bd.df_merge2015
df_merge2016 = bd.df_merge2016

#### Tratamiento de nulos en varible respuesta y variables de la tabla retirados

df_merge2015['Attrition'] = df_merge2015['Attrition'].fillna('No')
df_merge2016['Attrition'] = df_merge2016['Attrition'].fillna('No')
df_merge2015['retirementDate'] = df_merge2015['retirementDate'].fillna('No aplica')
df_merge2016['retirementDate'] = df_merge2016['retirementDate'].fillna('No aplica')

######### Encoder para la variable respuesta

# Inicializa el LabelEncoder
encoder = LabelEncoder()

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

list_dummies = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole','MaritalStatus']


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
m_rf= RandomForestClassifier(random_state=42,class_weight='balanced')
m_gbc = GradientBoostingClassifier()

modelos=list([m_lreg,m_rtree, m_rf, m_gbc])

var_names=func.sel_variables(modelos,X_train,Y,threshold="2*mean")
var_names.shape

X2=X_train[var_names] ### matriz con variables seleccionadas
X2.info()  #### Variables seleccionadas 
X_train.info() ##### Todas las variables

##### Seleccion del mejor modelo probando con todas las variables y las seleccionadas

acc_X_train = func.medir_modelos(modelos,"f1",X_train,Y,21) ## score con la base con todas las variables
acc_X2= func.medir_modelos(modelos,"f1",X2,Y,21) ### score con la base con variables seleccionadas

acc=pd.concat([acc_X_train,acc_X2],axis=1)
acc.columns=['m_lreg', 'm_dtree', 'm_rf','m_gbc','lreg_sel','rtree_sel', 'rf_sel','gbc_sel']

acc_X_train.plot(kind='box') #### gráfico para modelos todas las varibles
plt.show()

acc_X2.plot(kind='box') ### gráfico para modelo variables seleccionadas
plt.show()

acc.plot(kind='box') ### gráfico para modelos sel y todas las variables
plt.show()

acc.mean()

####### Afinamiento de hiperparametros

# Definición de cuadricula de hiperparametros para RandomForest

parameters = {'n_estimators': [3, 500, 100], 'max_features': [5, 20], 'min_samples_split': [100, 20, 5]}

##n_estimator numero de arboles de decisiones
##max feratures numero de variables analizadas para la participacion de nodos del arbol
## min_samples_split: mínimo de observaciones que pueden quedar en un nodo

tun_rf=RandomizedSearchCV(m_rf,param_distributions=parameters,n_iter=15,scoring="f1")
tun_rf.fit(X2,Y)

pd.set_option('display.max_colwidth', 100)
resultados=tun_rf.cv_results_
tun_rf.best_params_
pd_resultados=pd.DataFrame(resultados)
pd_resultados[["params","mean_test_score"]].sort_values(by="mean_test_score", ascending=False)

# Definición de cuadricula de hiperparametros para arboles de decisión

param_grid = { 'max_depth': np.arange(1, 21), 'criterion': ['gini', 'entropy']}

## cv es el número de divisiones para la validación cruzada, scoring es la métrica que deseas optimizar (en este caso, f1), y n_jobs es el número de núcleos de CPU que deseas utilizar para la búsqueda en paralelo.

grid_search = GridSearchCV(m_rtree, param_grid, cv=10, scoring='f1', n_jobs=20)

grid_search.fit(X_train, Y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)

#### Como se puede observar el modelo RandomForest se desempeña mucho mejor que el resto de modelos incluso despues del afinamiento de hiperparametros

rf_final=tun_rf.best_estimator_ ### Guardar el modelo con hyperparameter tunning
m_rtree=m_rtree.fit(X2,Y)

###### evaluar modelos afinados finales ###########

#####Evaluar métrica de entrenamiento y evaluación para mirar sobre ajuste ####

eval=cross_validate(rf_final,X2,Y,cv=30,scoring="f1",return_train_score=True)
eval2=cross_validate(m_rtree,X2,Y,cv=30,scoring="f1",return_train_score=True)

#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para RF
train_rf=pd.DataFrame(eval['train_score'])
test_rf=pd.DataFrame(eval['test_score'])
train_test_rf=pd.concat([train_rf, test_rf],axis=1)
train_test_rf.columns=['train_score','test_score']

#### convertir resultado de evaluacion entrenamiento y evaluacion en data frame para arboles de decisión
train_dtree=pd.DataFrame(eval2['train_score'])
test_dtree=pd.DataFrame(eval2['test_score'])
train_test_dtree=pd.concat([train_dtree, test_dtree],axis=1)
train_test_dtree.columns=['train_score','test_score']

train_test_dtree["test_score"].mean()
train_test_rf["test_score"].mean()

##### Mirar importancia de variables para tomar acciones ###

importancia = rf_final.feature_importances_
importancia = pd.DataFrame(importancia, columns=['Importancia'])

X3 = pd.DataFrame(X2.columns, columns=['Variables'])

# Concatenar las Series al DataFrame X2
X2_con_importancias = pd.concat([X3, importancia], axis=1)

X2_con_importancias = X2_con_importancias.sort_values(by=['Importancia'], ascending=False)

# Matriz de confusión
# ==============================================================================
y_hat=rf_final.predict(X2)
fig = plt.figure(figsize=(11,11))
cm = confusion_matrix(Y,y_hat, labels=rf_final.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=rf_final.classes_)
disp.plot(cmap='gist_earth')
plt.show()

### función para exportar y guardar objetos de python (cualqueira)

joblib.dump(rf_final, "rf_final.pkl") ##
joblib.dump(m_rtree, "m_dtree.pkl") ##
joblib.dump(columnas_a_escalar, "list_cat.pkl") ### para realizar imputacion variables
joblib.dump(list_dummies, "list_dummies.pkl") ### para convertir a dummies
joblib.dump(var_names, "var_names.pkl") ### para variables con que se entrena modelo
joblib.dump(scaler, "scaler.pkl") ##

### funcion para cargar objeto guardado ###

rf_final = joblib.load("rf_final.pkl")
m_dtree = joblib.load("m_dtree.pkl")
ist_cat=joblib.load("list_cat.pkl")
list_dummies=joblib.load("list_dummies.pkl")
var_names=joblib.load("var_names.pkl")
scaler=joblib.load("scaler.pkl")