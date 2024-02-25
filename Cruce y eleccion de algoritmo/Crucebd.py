import CodigoVS as basegen
import tratamientobdsatisfaccion as basesat
import sys ## saber ruta de la que carga paquetes
import pandas as pd


###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Trabajo practico\\CasoEstudioHR\\Tratamiento-exploracion') ## este comanda agrega una ruta

general = basegen.df_general
satisfaccion = basesat.df_filled

df_merge = pd.merge(general, satisfaccion, left_on='EmployeeID', right_on='EmployeeID')

df_merge.columns