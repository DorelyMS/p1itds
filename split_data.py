
from sklearn.model_selection import train_test_split

def split_aleatorio(variables_input,variable_output,porcentaje_prueba, semilla):
    X_train, X_test, y_train, y_test = train_test_split(variables_input, variable_output, test_size=porcentaje_prueba, random_state=semilla)
    return X_train, X_test, y_train, y_test


def split_tiempo(archivo,campo_criterio,anio_mes):
    archivo_2=archivo.loc[archivo[campo_criterio]<=anio_mes]
    archivo_3=archivo.loc[archivo[campo_criterio]>anio_mes]
    X_train=archivo_2.iloc[:, :-1]
    y_train=archivo_2.iloc[:, -1]
    X_test=archivo_3.iloc[:, :-1]
    y_test=archivo_3.iloc[:, -1]
    return X_train, X_test, y_train, y_test


    
    