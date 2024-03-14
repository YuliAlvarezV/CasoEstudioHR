import funciones as func  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np
import variables as var
import sys ## saber ruta de la que carga paquetes
import matplotlib.pyplot as plt ### gráficos
import Crucebd as cruce

sys.path
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\escalado-eleccion-var') ## este comanda agrega una ruta
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\Cruce-eleccion.algoritmos') ## este comanda agrega una ruta

###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":
    ####Se trae la base de datos de 2016 para las predicciones de 2017, teniendo en cuenta que este archivo ya fue tratado anteriormente
    df_t= var.X_test[var_names]

    ##Cargar modelo y predecir
    rf_final = joblib.load("rf_final.pkl")
    predicciones=rf_final.predict(df_t)
    pd_pred=pd.DataFrame(predicciones, columns=['attrition_2017'])

    ###Crear base con predicciones ####

    idempleados = cruce.df_merge2016['EmployeeID']
    perf_pred=pd.concat([idempleados,df_t,pd_pred],axis=1)

    #### cantidad de empleados que segun la prediccion se retiraran

    retirados=perf_pred['attrition_2017'].value_counts()
    # Crear un gráfico de pastel
    plt.figure(figsize=(8, 8))  # Tamaño del gráfico
    plt.pie(retirados, labels= retirados.index, autopct='%1.1f%%', startangle=140)  # Crear gráfico de pastel
    plt.axis('equal')  # Aspecto de círculo
    plt.title("Cantidad de retiradods predichos para 2017")  # Título del gráfico

    plt.show()

    ####ver_predicciones_bajas ###
    emp_pred_bajo=perf_pred.sort_values(by=["attrition_2017"],ascending=True).head(10)
    
    emp_pred_bajo.set_index('EmployeeID', inplace=True) 
    pred=emp_pred_bajo.T
    
    importancia1 = rf_final.feature_importances_   #importancia de las variables seleccionadas
    importancia1 = pd.DataFrame(importancia1, columns=['Importancia'])
    X3 = pd.DataFrame(df_t.columns, columns=['Variables'])

    # Concatenar las Series al DataFrame df_t
    variables_con_importancias = pd.concat([X3, importancia1], axis=1)
    
    pred.to_excel("prediccion.xlsx")   #### exportar predicciones mas bajas y variables explicativas
    variables_con_importancias.to_excel("variables_con_importancias.xlsx") ### exportar importancias de las variables para analizar predicciones