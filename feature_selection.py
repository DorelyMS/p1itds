#Este módulo contiene las funciones utilizadas para feature selection

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier


def chi_squared(variables_input,variable_output,p_value):
    qal75=variables_input[variable_output].quantile(.75)
    def selecciona_variables_categoricas(variables_input):
        lista_variables_categoricas=variables_input.select_dtypes(include = 'object').columns.values
        archivo_variables_categoricas = variables_input.loc[:, lista_variables_categoricas]
        return archivo_variables_categoricas
    archivocat=selecciona_variables_categoricas(variables_input)
    archivocat=pd.get_dummies(archivocat)
    respuestas=archivocat
    #se calculan niveles de clasificación para las variables categóricas. consumo_alto=0 significa consumo>=q75
    predictor=np.where(variables_input[variable_output]>=qal75,1,0)
    #chi_scores genera 2 arrays, el primero representa chi square values y el segundo los p-values
    chi_scores = chi2(respuestas,predictor)
    #p_valuesalfa es una tabla con la relación de variables significativas y no significativas para el predictor
    p_valuesalfa = pd.DataFrame(chi_scores[1],index = respuestas.columns)
    p_valuesalfa.rename(columns = {0:'p-values'}, inplace = True)
    p_valuesalfa['significancia'] = np.where(p_valuesalfa['p-values']<p_value, 'significativa', 'no significativa')
    lista_significativas=p_valuesalfa.loc[p_valuesalfa.significancia == 'significativa'].index
    return pd.DataFrame(lista_significativas)

def function(variables_input,variable_output,p_value):
    
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
    
    #Regresion lineal
    n=variables_input.shape[1]
    nombres_variables=np.array(variables_input.columns.values)
    nombres_variables=np.append(['intercepto'],nombres_variables)
    X = sm.add_constant(variables_input.iloc[:,0:n].values.reshape(variables_input.shape[0],n))
    modelo= sm.OLS(variable_output, X)
    modelo2 = modelo.fit()
    p_values = modelo2.summary2().tables[1]['P>|t|']
    
    while max(p_values)>p_value:
        for i in range(0,len(p_values)):
            if p_values[i]>p_value and p_values[i]==max(p_values):
                #variables_input.drop(variables_input.columns[i-1], axis=1,inplace=True)
                X=np.delete(X, i, 1)
                nombres_variables=np.delete(nombres_variables, i)
                #n=n-1
        #X2 = sm.add_constant(x.iloc[:,0:n].values.reshape(x.shape[0],n))
        modelo = sm.OLS(variable_output, X)
        modelo2 = modelo.fit()
        p_values = modelo2.summary2().tables[1]['P>|t|']
        
    return pd.DataFrame(nombres_variables)

#variables_input.head()
#pd.DataFrame(nombres_variables)




def RandomForest(variables_input,variable_output,numero_arboles,semilla):
    
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
    #sfm = SelectFromModel(clf, threshold=umbral)
    clf.fit(variables_input_array, variable_output)
    for feature in zip(nombres_columnas, clf.feature_importances_):
        print(feature)

