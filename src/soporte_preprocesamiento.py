# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys 

import warnings
warnings.filterwarnings("ignore")




# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor



from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler




sys.path.append(os.path.abspath("src"))   
import soporte_preprocesamiento as f
# import plotly_express as px


# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon
from scipy import stats
# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product, combinations


from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Para la codificación de las variables numéricas
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder # para poder aplicar los métodos de OneHot, Ordinal,  Label y Target Encoder 






def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    # for categoria in dataframe[columna_control].unique():
    #     dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas categóricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe(include = "O").T)
        
    #     print("\n ..................... \n")
    #     print(f"Los principales estadísticos de las columnas numéricas para el {categoria} son: ")
    #     display(dataframe_filtrado.describe().T)


def separar_dataframe(df):
    return df.select_dtypes(include=np.number), df.select_dtypes(include="O")

def plot_numericas(df, tamanio):
    cols_numericas=df.columns

    nfilas=math.ceil(len(cols_numericas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat
    for indice,col in enumerate(cols_numericas):
        sns.histplot(x=col, data= df, ax= axes[indice], bins=50)
        axes[indice].set_title(col)
        axes[indice].set_xlabel("")


    if len(cols_numericas) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()
    
def plot_categoricas(df, tamanio, paleta="mako"):
    cols_categoricas=df.columns

    nfilas=math.ceil(len(cols_categoricas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat
    for indice,col in enumerate(cols_categoricas):
        sns.countplot(x=col, data= df, ax= axes[indice], palette=paleta, order= df[col].value_counts().index)
        axes[indice].set_title(col)
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation=90)
        plt.tight_layout() 


    if len(cols_categoricas) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass

    plt.tight_layout()    


def relacion_dependiente_categoricas(df, variable_dependiente, tamanio=(15,8), paleta="mako"):
    df_cat= f.separar_dataframe(df)[1]
    cols_categoricas=df_cat.columns
    nfilas=math.ceil(len(cols_categoricas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat

    for indice, col in enumerate(cols_categoricas):
        datos_agrupados= df.groupby(col)[variable_dependiente].mean().reset_index().sort_values(variable_dependiente, ascending=False)
        sns.barplot(x=col ,y=variable_dependiente, data=datos_agrupados, palette=paleta, ax=axes[indice])
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
        axes[indice].set_xlabel("")
    plt.tight_layout()



def relacion_dependiente_numericas(df, variable_dependiente, tamanio=(15,8), paleta="mako"):
    df_numericas= f.separar_dataframe(df)[0]
    cols_numericas=df_numericas.columns
    nfilas=math.ceil(len(cols_numericas) /2)  #es para que si da 4,5, se quede con 5 y no con 4 porque de normal redondea hacia abajo
    fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
    axes=axes.flat

    for indice, col in enumerate(cols_numericas):
        if col == variable_dependiente:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x=col ,y=variable_dependiente, data=df_numericas, palette=paleta, ax=axes[indice])

            axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
            axes[indice].set_xlabel("")
    plt.tight_layout()


def matriz_correlacion(df,tamanio=(10,7)):
    matriz_corr=df.corr(numeric_only=True)
    plt.figure(figsize=tamanio)
    mascara=np.triu(np.ones_like(matriz_corr, dtype=np.bool_))
    sns.heatmap(matriz_corr, annot=True, vmin=-1, vmax=1, mask=mascara)
    plt.tight_layout()



#OUTLIERS


def detectar_outliers(df, tamanio=(15,8), color="orange"):
    df_numericas=f.separar_dataframe(df)[0]
    num_filas=math.ceil(len(df_numericas.columns)/2)
    fig, axes = plt.subplots(nrows=num_filas,ncols=2, figsize=tamanio)
    axes= axes.flat
    for indice, col in enumerate(df_numericas.columns):
        sns.boxplot(x= col, data= df_numericas, ax=axes[indice], color=color)
        axes[indice].set_title(f"Outliers de {col}")
        axes[indice].set_xlabel("")
    if len(df_numericas.columns) %2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    

    plt.tight_layout()




def identificar_outliers_iqr(df, k=1.5):
    df_num=df.select_dtypes(include=np.number)
    dicc_outliers={}
    for columna in df_num.columns:
        Q1, Q3= np.nanpercentile(df[columna], (25, 75))
        iqr= Q3 - Q1
        limite_superior= Q3 + (iqr * k)
        limite_inferior= Q1 - (iqr * k)

        condicion_sup= df[columna] > limite_superior
        condicion_inf= df[columna] < limite_inferior
        df_outliers= df[condicion_inf | condicion_sup]
        print(f"La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers")
        if not df_outliers.empty:
            dicc_outliers[columna]= df_outliers

    return dicc_outliers    



def plot_outliers_univariados(df, tipo_grafica, tamanio,bins=20, whis=1.5):
    df_num=df.select_dtypes(include=np.number)

    fig, axes = plt.subplots(nrows= math.ceil(len(df_num.columns)/ 2), ncols=2, figsize= tamanio)
    axes=axes.flat

    for indice, columna in enumerate(df_num.columns):
        if tipo_grafica.lower() =="h":
            sns.histplot(x= columna, data= df, ax= axes[indice], bins= bins)
        elif tipo_grafica.lower() == "b":
            sns.boxplot(x= columna, data= df, ax= axes[indice], whis=whis, flierprops={"markersize" : 4, "markerfacecolor": "red"})
        else:
            print("No has elegido una grafica correcta elige h o b")
        axes[indice].set_title(f"Distribución columna {columna}")
        axes[indice].set_xlabel("")

    if len(df_num.columns) % 2 !=0:
        fig.delaxes(axes[-1])

    plt.tight_layout()



def identificar_outliers_zscore(df, limite_desviaciones =3):
    df_num=df.select_dtypes(include=np.number)
    dicc_outliers={}
    for columna in df_num.columns:
        condicion_zscore= abs(zscore(df[columna])) >= limite_desviaciones
        df_outliers= df[condicion_zscore]
        print(f"La cantidad de outliers para la {columna.upper()} es de {df_outliers.shape[0]} outliers")
        if not df_outliers.empty:
            dicc_outliers[columna]= df_outliers
    return dicc_outliers


#ENCODING


def visualizar_categoricas(df, dependiente, tamanio, tipo_graf="box", bigote=1.5, metrica="mean"):
    df_categoricas=f.separar_dataframe(df)[1]
    num_filas=math.ceil(len(df_categoricas.columns)/2)

    fig, axes = plt.subplots(nrows= num_filas, ncols= 2, figsize= tamanio)
    axes= axes.flat

    for indice, col in enumerate(df_categoricas.columns):
        if tipo_graf.lower()== "box":
            sns.boxplot(x= col, y = dependiente, data= df, whis = bigote, hue=col, legend=False, ax= axes[indice])

        elif tipo_graf.lower()== "bar":
            sns.barplot(x=col, y = dependiente, ax= axes[indice], data= df, estimator=metrica, palette="mako")

        else:
            print("No has eleigido una grafica correcta elige box o bar")
        axes[indice].set_title(f"Relación columna {col}, con {dependiente}")
        axes[indice].set_xlabel("")
        plt.tight_layout()






class Asunciones:
    def __init__(self, dataframe, columna_numerica):

        self.dataframe = dataframe
        self.columna_numerica = columna_numerica
        
    

    def identificar_normalidad(self, metodo='shapiro', alpha=0.05, verbose=True):
        """
        Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk o Kolmogorov-Smirnov.

        Parámetros:
            metodo (str): El método a utilizar para la prueba de normalidad ('shapiro' o 'kolmogorov').
            alpha (float): Nivel de significancia para la prueba.
            verbose (bool): Si se establece en True, imprime el resultado de la prueba. Si es False, Returns el resultado.

        Returns:
            bool: True si los datos siguen una distribución normal, False de lo contrario.
        """

        if metodo == 'shapiro':
            _, p_value = stats.shapiro(self.dataframe[self.columna_numerica])
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Shapiro-Wilk." if resultado else f"los datos no siguen una distribución normal según el test de Shapiro-Wilk."
        
        elif metodo == 'kolmogorov':
            _, p_value = stats.kstest(self.dataframe[self.columna_numerica], 'norm')
            resultado = p_value > alpha
            mensaje = f"los datos siguen una distribución normal según el test de Kolmogorov-Smirnov." if resultado else f"los datos no siguen una distribución normal según el test de Kolmogorov-Smirnov."
        else:
            raise ValueError("Método no válido. Por favor, elige 'shapiro' o 'kolmogorov'.")

        if verbose:
            print(f"Para la columna {self.columna_numerica}, {mensaje}")
        else:
            return resultado

        
    def identificar_homogeneidad (self,  columna_categorica):
        
        """
        Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.

        Parámetros:
        - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
        - columna_categorica (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.

        Returns:
        No Returns nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
        Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
        se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
        """
        
        # lo primero que tenemos que hacer es crear tantos conjuntos de datos para cada una de las categorías que tenemos, Control Campaign y Test Campaign
        valores_evaluar = []
        
        for valor in self.dataframe[columna_categorica].unique():
            valores_evaluar.append(self.dataframe[self.dataframe[columna_categorica]== valor][self.columna_numerica])

        statistic, p_value = stats.levene(*valores_evaluar)
        if p_value > 0.05:
            print(f"En la variable {columna_categorica} las varianzas son homogéneas entre grupos.")
        else:
            print(f"En la variable {columna_categorica} las varianzas NO son homogéneas entre grupos.")




class TestEstadisticos:
    def __init__(self, dataframe, variable_respuesta, columna_categorica):
        """
        Inicializa la instancia de la clase TestEstadisticos.

        Parámetros:
        - dataframe: DataFrame de pandas que contiene los datos.
        - variable_respuesta: Nombre de la variable respuesta.
        - columna_categorica: Nombre de la columna que contiene las categorías para comparar.
        """
        self.dataframe = dataframe
        self.variable_respuesta = variable_respuesta
        self.columna_categorica = columna_categorica

    def generar_grupos(self):
        """
        Genera grupos de datos basados en la columna categórica.

        Retorna:
        Una lista de nombres de las categorías.
        """
        lista_categorias =[]
    
        for value in self.dataframe[self.columna_categorica].unique():
            variable_name = value  # Asigna el nombre de la variable
            variable_data = self.dataframe[self.dataframe[self.columna_categorica] == value][self.variable_respuesta].values.tolist()
            globals()[variable_name] = variable_data  
            lista_categorias.append(variable_name)
    
        return lista_categorias

    def comprobar_pvalue(self, pvalor):
        """
        Comprueba si el valor p es significativo.

        Parámetros:
        - pvalor: Valor p obtenido de la prueba estadística.
        """
        if pvalor < 0.05:
            print("Hay una diferencia significativa entre los grupos")
        else:
            print("No hay evidencia suficiente para concluir que hay una diferencia significativa.")

    def test_manwhitneyu(self, categorias): # SE PUEDE USAR SOLO PARA COMPARAR DOS GRUPOS, PERO NO ES NECESARIO QUE TENGAN LA MISMA CANTIDAD DE VALORES
        """
        Realiza el test de Mann-Whitney U.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.mannwhitneyu(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Mann-Whitney U:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def test_wilcoxon(self, categorias): # SOLO LO PODEMOS USAR SI QUEREMOS COMPARAR DOS CATEGORIAS Y SI TIENEN LA MISMA CANTIDAD DE VALORES 
        """
        Realiza el test de Wilcoxon.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.wilcoxon(*[globals()[var] for var in categorias])

        print("Estadístico del Test de Wilcoxon:", statistic)
        print("Valor p:", p_value)

        # Imprime el estadístico y el valor p
        print("Estadístico de prueba:", statistic)
        print("Valor p:", p_value) 

        self.comprobar_pvalue(p_value)

    def test_kruskal(self, categorias):
       """
       Realiza el test de Kruskal-Wallis.

       Parámetros:
       - categorias: Lista de nombres de las categorías a comparar.
       """
       statistic, p_value = stats.kruskal(*[globals()[var] for var in categorias])

       print("Estadístico de prueba:", statistic)
       print("Valor p:", p_value)

       self.comprobar_pvalue(p_value)

    
    def test_anova(self, categorias):
        """
        Realiza el test ANOVA.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        statistic, p_value = stats.f_oneway(*[globals()[var] for var in categorias])

        print("Estadístico F:", statistic)
        print("Valor p:", p_value)

        self.comprobar_pvalue(p_value)

    def post_hoc(self):
        """
        Realiza el test post hoc de Tukey.
        
        Retorna:
        Un DataFrame con las diferencias significativas entre los grupos.
        """
        resultado_posthoc =  pairwise_tukeyhsd(self.dataframe[self.variable_respuesta], self.dataframe[self.columna_categorica])
        tukey_df =  pd.DataFrame(data=resultado_posthoc._results_table.data[1:], columns=resultado_posthoc._results_table.data[0])
        tukey_df['group_diff'] = tukey_df['group1'] + '-' + tukey_df['group2']
        return tukey_df[['meandiff', 'p-adj', 'lower', 'upper', 'group_diff']]
    #[tukey_df["p-adj"] <= 0.05]

    def run_all_tests(self):
        """
        Ejecuta todos los tests estadísticos disponibles en la clase.

        Parámetros:
        - categorias: Lista de nombres de las categorías a comparar.
        """
        print("Generando grupos...")
        categorias_generadas = self.generar_grupos()
        print("Grupos generados:", categorias_generadas)


        test_methods = {
            "mannwhitneyu": self.test_manwhitneyu,
            "wilcoxon": self.test_wilcoxon,
            "kruskal": self.test_kruskal,
            "anova": self.test_anova
        }

        test_choice = input("¿Qué test desea realizar? (mannwhitneyu, wilcoxon, kruskal, anova): ").strip().lower()
        test_method = test_methods.get(test_choice)
        if test_method:
            print(f"\nRealizando test de {test_choice.capitalize()}...")
            test_method(categorias_generadas)
        else:
            print("Opción de test no válida.")
        
        print("Los resultados del test de Tukey son: \n")
        display(self.post_hoc())




class Encoding:
    """
    Clase para realizar diferentes tipos de codificación en un DataFrame.

    Parámetros:
        - dataframe: DataFrame de pandas, el conjunto de datos a codificar.
        - diccionario_encoding: dict, un diccionario que especifica los tipos de codificación a realizar.
        - variable_respuesta: str, el nombre de la variable objetivo.

    Métodos:
        - one_hot_encoding(): Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.
        - get_dummies(prefix='category', prefix_sep='_'): Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.
        - ordinal_encoding(): Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.
        - label_encoding(): Realiza codificación label en las columnas especificadas en el diccionario de codificación.
        - target_encoding(): Realiza codificación target en la variable especificada en el diccionario de codificación.
        - frequency_encoding(): Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.
    """

    def __init__(self, dataframe, diccionario_encoding, variable_respuesta):
        self.dataframe = dataframe
        self.diccionario_encoding = diccionario_encoding
        self.variable_respuesta = variable_respuesta
    
    def one_hot_encoding(self):
        """
        Realiza codificación one-hot en las columnas especificadas en el diccionario de codificación.

        Returns:
            - dataframe: DataFrame de pandas, el DataFrame con codificación one-hot aplicada.
        """
        # accedemos a la clave de 'onehot' para poder extraer las columnas a las que que queramos aplicar OneHot Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("onehot", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase de OneHot
            one_hot_encoder = OneHotEncoder()

            # transformamos los datos de las columnas almacenadas en la variable col_code
            trans_one_hot = one_hot_encoder.fit_transform(self.dataframe[col_encode])

            # el objeto de la transformación del OneHot es necesario convertilo a array (con el método toarray()), para luego convertilo a DataFrame
            # además, asignamos el valor de las columnas usando el método get_feature_names_out()
            oh_df = pd.DataFrame(trans_one_hot.toarray(), columns=one_hot_encoder.get_feature_names_out())

            # concatenamos los resultados obtenidos en la transformación con el DataFrame original
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)
    
        return self.dataframe
    
    def get_dummies(self, prefix='category', prefix_sep="_"):
        """
        Realiza codificación get_dummies en las columnas especificadas en el diccionario de codificación.

        Parámetros:
        - prefix: str, prefijo para los nombres de las nuevas columnas codificadas.
        - prefix_sep: str, separador entre el prefijo y el nombre original de la columna.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación get_dummies aplicada.
        """
        # accedemos a la clave de 'dummies' para poder extraer las columnas a las que que queramos aplicar este método. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("dummies", [])

        if col_encode:
            # aplicamos el método get_dummies a todas las columnas seleccionadas, y determinamos su prefijo y la separación
            df_dummies = pd.get_dummies(self.dataframe[col_encode], dtype=int, prefix=prefix, prefix_sep=prefix_sep)
            
            # concatenamos los resultados del get_dummies con el DataFrame original
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
            
            # eliminamos las columnas original que ya no nos hacen falta
            self.dataframe.drop(col_encode, axis=1, inplace=True)
    
        return self.dataframe

    def ordinal_encoding(self):
        """
        Realiza codificación ordinal en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación ordinal aplicada.
        """

        # Obtenemos las columnas a codificar bajo la clave 'ordinal'. Si no existe la clave, la variable col_encode será una lista vacía.
        col_encode = self.diccionario_encoding.get("ordinal", {})

        # Verificamos si hay columnas a codificar.
        if col_encode:

            # Obtenemos las categorías de cada columna especificada para la codificación ordinal.
            orden_categorias = list(self.diccionario_encoding["ordinal"].values())
            
            # Inicializamos el codificador ordinal con las categorías especificadas.
            ordinal_encoder = OrdinalEncoder(categories=orden_categorias, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)
            
            # Aplicamos la codificación ordinal a las columnas seleccionadas.
            ordinal_encoder_trans = ordinal_encoder.fit_transform(self.dataframe[col_encode.keys()])

            # Eliminamos las columnas originales del DataFrame.
            self.dataframe.drop(col_encode, axis=1, inplace=True)
            
            # Creamos un nuevo DataFrame con las columnas codificadas y sus nombres.
            ordinal_encoder_df = pd.DataFrame(ordinal_encoder_trans, columns=ordinal_encoder.get_feature_names_out())

            # Concatenamos el DataFrame original con el DataFrame de las columnas codificadas.
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), ordinal_encoder_df], axis=1)

        return self.dataframe


    def label_encoding(self):
        """
        Realiza codificación label en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación label aplicada.
        """

        # accedemos a la clave de 'label' para poder extraer las columnas a las que que queramos aplicar Label Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("label", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase LabelEncoder()
            label_encoder = LabelEncoder()

            # aplicamos el Label Encoder a cada una de las columnas, y creamos una columna con el nombre de la columna (la sobreescribimos)
            self.dataframe[col_encode] = self.dataframe[col_encode].apply(lambda col: label_encoder.fit_transform(col))
     
        return self.dataframe

    def target_encoding(self):
        """
        Realiza codificación target en la variable especificada en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación target aplicada.
        """
        
        # accedemos a la clave de 'target' para poder extraer las columnas a las que que queramos aplicar Target Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("target", [])

        # si hay contenido en la lista 
        if col_encode:

            # instanciamos la clase 
            target_encoder = TargetEncoder(smooth="auto")

            # transformamos los datos de las columnas almacenadas en la variable col_code y añadimos la variable respuesta para que calcule la media ponderada para cada categória de las variables
            target_encoder_trans = target_encoder.fit_transform(self.dataframe[[self.variable_respuesta]], self.dataframe[col_encode])
            
            # creamos un DataFrame con los resultados de la transformación
            target_encoder_df = pd.DataFrame(target_encoder_trans, columns=target_encoder.get_feature_names_out())

            # eliminamos las columnas originales
            self.dataframe.drop(col_encode, axis=1, inplace=True)

            # concatenamos los dos DataFrames
            self.dataframe = pd.concat([self.dataframe.reset_index(drop=True), target_encoder_df], axis=1)

        return self.dataframe

    def frequency_encoding(self):
        """
        Realiza codificación de frecuencia en las columnas especificadas en el diccionario de codificación.

        Returns:
        - dataframe: DataFrame de pandas, el DataFrame con codificación de frecuencia aplicada.
        """

        # accedemos a la clave de 'frequency' para poder extraer las columnas a las que que queramos aplicar Frequency Encoding. En caso de que no exista la clave, esta variable será una lista vacía
        col_encode = self.diccionario_encoding.get("frequency", [])

        # si hay contenido en la lista 
        if col_encode:

            # iteramos por cada una de las columnas a las que les queremos aplicar este tipo de encoding
            for categoria in col_encode:

                # calculamos las frecuencias de cada una de las categorías
                frecuencia = self.dataframe[categoria].value_counts(normalize=True)

                # mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
                self.dataframe[categoria] = self.dataframe[categoria].map(frecuencia)
        
        return self.dataframe
    





def extract_parking_info(parking_info):
    if parking_info== "desconocido":  # Si es NaN
        return 'desconocido'
    try:
        
        info_dict = eval(parking_info)
        
        return info_dict.get('isParkingSpaceIncludedInPrice', 'desconocido')
    except Exception:
        return 'desconocido'
    

def escalador_datos(data, columns, method = "robust"):
    
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "norm":
        scaler =  Normalizer()
    df_scaled = pd.DataFrame(scaler.fit_transform(data[columns]), columns=[c+"_"+method for c in columns])  
    df_concat=pd.concat([data,df_scaled], axis=1)  
    return df_concat, scaler