#!/usr/bin/env python
# coding: utf-8

# In[225]:


import load_data
import clean_data
import transform_data
import eda
import split_data

datos=load_data.carga_archivo('interrupcion-legal-del-embarazo.csv')
load_data.observaciones_variables(datos)


# In[226]:


datos.sample(6)


# In[230]:


#limpiamos los nombres de las columnas
clean_data.estandariza_variables(datos)


# In[231]:


transform_data.tipo_variables(datos)


# In[232]:


#Limpiamos los datos

transform_data.cambiar_minusculas_variable(datos,'mes')
transform_data.cambiar_minusculas_variable(datos,'autoref')
transform_data.cambiar_minusculas_variable(datos,'edocivil_descripcion')
transform_data.cambiar_minusculas_variable(datos,'desc_derechohab')
transform_data.cambiar_minusculas_variable(datos,'nivel_edu')
transform_data.cambiar_minusculas_variable(datos,'ocupacion')
transform_data.cambiar_minusculas_variable(datos,'religion')
transform_data.cambiar_minusculas_variable(datos,'parentesco')
transform_data.cambiar_minusculas_variable(datos,'entidad')
transform_data.cambiar_minusculas_variable(datos,'alc_o_municipio')
transform_data.cambiar_minusculas_variable(datos,'consejeria')
transform_data.cambiar_minusculas_variable(datos,'anticonceptivo')
transform_data.cambiar_minusculas_variable(datos,'motiles')
transform_data.cambiar_minusculas_variable(datos,'h_fingreso')
transform_data.cambiar_minusculas_variable(datos,'desc_servicio')
transform_data.cambiar_minusculas_variable(datos,'p_consent')
transform_data.cambiar_minusculas_variable(datos,'procile')
transform_data.cambiar_minusculas_variable(datos,'s_complica')
transform_data.cambiar_minusculas_variable(datos,'panticoncep')
transform_data.quitar_acentos(datos)


# In[233]:


datos.sample(6)


# In[234]:


#Cambiamos los tipos de datos adecuados, creamos la variable target
import numpy as np

transform_data.cambiar_tipo_variable(datos,'cve_hospital','str')
datos['target']=np.where(datos['edad']>=23,1,0)
datos['targetdesc']=np.where(datos['edad']>=23,">=23 ","<=22")


# In[8]:


#Generamos el data profiling

import pandas_profiling

datos.profile_report(style={'full_width':True})


# In[235]:


#Estandarizamos variables para EDA

def estandariza_ocupacion(trabajo):
    if trabajo=="estudiante":
        return "estudiante"
    elif trabajo=="ama de casa":
        return "ama de casa"
    elif trabajo=="desempleada":
        return "desempleada"
    elif trabajo is None:
        return np.nan
    else:
        return "empleada"

def estandariza_educacion(educacion):
    if educacion=="primaria completa":
        return "primaria completa"
    elif educacion=="secundaria completa":
        return "secundaria completa"
    elif educacion=="preparatoria completa":
        return "preparatoria completa"
    elif educacion=="licenciatura completa":
        return "licenciatura completa"
    elif educacion=="posgrado" or educacion=="maestria" or educacion=="doctorado":
        return "posgrado completo"
    elif educacion==None:
        pass
    else:
        return "otro"
    
def estandariza_embarazos(gestaciones):
    if gestaciones==0:
        return 1
    else:
        return gestaciones

def estandariza_edocivil(edocivil):
    if edocivil=="n/e":
        return np.nan
    else:
        return edocivil

datos['ocupacion_std'] = datos.apply(lambda x: estandariza_ocupacion(x.ocupacion), axis=1)
datos['educacion_std'] = datos.apply(lambda x: estandariza_educacion(x.nivel_edu), axis=1)
datos['embarazos_std'] = datos.apply(lambda x: estandariza_embarazos(x.gesta), axis=1)
datos['edocivil_std'] = datos.apply(lambda x: estandariza_edocivil(x.edocivil_descripcion), axis=1)


# In[236]:


datos.ocupacion.unique()


# In[ ]:


datos.educacion_std.unique()


# In[283]:


datos.nivel_edu.unique()


# In[238]:


datos.ocupacion_std.unique()


# In[239]:


datos.embarazos_std.unique()


# In[240]:


datos.edocivil_std.unique()


# **¿Cuáles son los 5 insights más importantes de este conjunto de datos**
# 
# * Observamos que las mujeres de 23 años en adelante, en promedio, pasaban al menos por su segundo embarazo cuando decidieron interrumpirlo legalmente.
# * Identificamos que aunque la edad promedio en la que se realiza un aborto legal por primera vez está alrededor de 24 años, este procedimiento con frecuencia se realiza en mujeres de cualquier edad (desde los 11 y hasta los 53 años).
# * Existe una fuerte reincidencia en la interrupción legal del embarazo por pacientes que ya se han practicado una o dos interrupciones previamente.
# * Las mujeres mayores a 23 años representan más del 70% de las empleadas, amas de casa y desempleadas. Mientras que en contraste, el 70% de las estudiantes que decidieron realizar la interrupción legal del embarazo son menores de 23 años.
# * La edad de inicio de vida sexual aparentemente se relacionada con el nivel de estudios completados. En promedio, las mujeres con estudios de licenciatura y posgrado terminados declaran haber iniciado su vida sexual cercanas a los 18 años mientras que las que sólo cuentan con primaria o secundaria lo hicieron alrededor de los 16 años. 

# In[241]:


from matplotlib import pyplot as plt

plt.figure(figsize=(15,8))
ax = sns.barplot(x="edad", y="gesta", data=datos)


# In[242]:


plt.figure(figsize=(15,8))
graf = sns.boxplot(x="nile", y="edad", data=datos, whis=np.inf)
graf = sns.stripplot(x="nile", y="edad", data=datos,jitter=True, linewidth=.5)
graf.set(xlabel="Interrupciones de embarazo previas", ylabel="Edades", title="Interrupciones de embarazo previas por edad")


# In[243]:


eda.grafico_barplot_orden_decreciente(datos,'ocupacion_std','target','Ocupación','% de mujeres >=23','Ocupación de mujeres 23+')


# In[244]:


eda.grafico_barplot_orden_decreciente(datos,'edocivil_std','target','Estado Civil''% de mujeres >=23','Ocupación de mujeres mayores')


# In[245]:


eda.grafico_barplot_orden_decreciente(datos,'educacion_std','fsexual','Educación','Edad inicio vida sexual','Estudios de mujeres mayores')


# In[246]:


datos.sample(6)


# In[247]:


m={
        'enero':'01',
        'febrero':'02',
        'marzo':'03',
        'abril':'04',
        'mayo':'05',
        'junio':'06',
        'julio':'07',
        'agosto':'08',
        'septiembre':'09',
        'octubre':'10',
        'noviembre':'11',
        'diciembre':'12'
    }

num_mes=np.array([m[x] for x in datos['mes'].values])

datos['mes_num']=num_mes


# In[248]:


datos['aniomes']=datos['ano'].astype(str)+datos['mes_num'].astype(str)
datos['aniomes']=datos['aniomes'].astype('int')


# In[271]:


columna_mover = ['target']
nuevo_orden = np.hstack((datos.columns.difference(columna_mover), columna_mover))
datos = datos.reindex(columns=nuevo_orden)
datos.head()


# In[276]:


datos_select=datos[['aniomes','edocivil_std','educacion_std','ocupacion_std','fsexual','nhijos','gesta','nile','target']]


# In[277]:


datos_select.sample(10)


# In[278]:


X_train, X_test, y_train, y_test=split_data.split_tiempo(datos_select,'aniomes',201809)


# In[279]:


transform_data.eliminar_variable(X_train,'aniomes')
transform_data.eliminar_variable(X_test,'aniomes')


# In[281]:


eda.tabla_estadisticos_descriptivos_variables_categoricas(X_train)


# In[ ]:





# In[265]:


######Este codigo usa random forest para calcular la importancia de variables

import pandas as pd
## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 

## This line instantiates the model. 
rf = RandomForestClassifier() 
## Fit the model on your training data.
rf.fit(X_train, y_train) 
## And score it on your testing data.
rf.score(X_test, y_test)

import pandas as pd

feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[ ]:




