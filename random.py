import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier

#También en arboles se tiene que estandarizar y eso???
#al aplicar esta funcion ya debe  estar separada en entrenamiento y prueba

def RandomForest(variables_input,variable_output,numero_arboles,semilla,umbral):
    
    #Imputa variables numéricas
    lista_variables_numericas=variables_input.select_dtypes(include = 'number').columns.values
    variables_estandarizar=variables_input.select_dtypes(include = 'number').values.reshape(variables_input.shape[0],len(lista_variables_numericas))
    imputar = SimpleImputer(strategy="mean")
    imputar.fit(variables_estandarizar)
    variables_input.loc[:, lista_variables_numericas] = imputar.transform(variables_estandarizar)
    
    #Estandariza variables numéricas
    variables_estandarizar=variables_input.select_dtypes(include = 'number').values.reshape(variables_input.shape[0],len(lista_variables_numericas))
    scaler = StandardScaler()
    variables_input.loc[:, lista_variables_numericas] = scaler.fit_transform(variables_estandarizar)    
    
    #One hot encoder
    lista_variables_categoricas=variables_input.select_dtypes(include = 'object').columns.values
    variables_input=pd.get_dummies(variables_input,columns=lista_variables_categoricas,prefix=lista_variables_categoricas)
    
    nombres_columnas=variables_input.columns.values
    variables_input_array=variables_input.values.reshape(variables_input.shape[0],variables_input.shape[1])
    clf = RandomForestClassifier(n_estimators=numero_arboles, random_state=semilla, n_jobs=-1)
    sfm = SelectFromModel(clf, threshold=umbral)
    sfm.fit(variables_input_array, variable_output)
    for feature in zip(nombres_columnas, sfm.feature_importances_):
        print(feature)
    