# Proyecto7-PrediccionCasas


## Entrenamiento del mejor modelo predictivo del precio del alquiler en Madrid
![Predicción de Alquileres en Madrid](https://github.com/jgilsu11/Proyecto7-PrediccionCasas/blob/main/Imagen/imagen%20alquiler.webp)  
  
  
***Descripción:***
El proyecto 7 consiste en la especificación e iteración de modelos predictivos hasta obtener un modleo predictivo óptimo para la predicción del alquiler de la vivienda en Madrid haciendo uso de archivos.py y Jupyter notebook.

Las técnicas usadas durante el proyecto son en su mayoría aquellas enseñadas durante la séptima semana de formación (Preprocesamiento(Gestión de nulos, Duplicados,Encoding, Estandarizacióny Gestión de Outliers) , generación y entrenamiento de modelos).

Adicionalmente, se usaron recursos obtenidos mediante research en documentación especializada, vídeos de YouTube e IA como motor de búsqueda y apoyo al aprendizaje.


***Estructura del Proyecto:***

El desarrollo del proyecto se gestionó de la siguiente manera:

- _En Primer lugar_, haciendo uso de JupyterNotebook como primer paso donde realizar ensayos con el código.  

- _En Segundo Lugar_, se creó una presentación basada en los datos.

- _Finalmente_, se realizó la documentación del proyecto en un archivo README (documento actual).

Por todo lo anterior, el usuario tiene acceso a:

        ├── datos/                                       # Donde se guardan los csv que se van generando en cada modelo 
        ├── Entorno/                                     # Entorno con todas las bibliotecas y versiones necesarias 
        ├── Imagen/                                      # Imagen para su uso en streamlit       
        ├── Modelos/                                     # Notebooks de Jupyter donde se han ido desarrollando los modelos
        ├── pickle_general/                              # Donde se han guardado los pickles para su uso en streamlit
        ├── src/                                         # Scripts (.py)
        ├── README.md                                    # Descripción del proyecto
        ├── web_final                                    # Script donde desarrollar streamlit (.py)          
        
***Requisitos e Instalación🛠️:***

Este proyecto usa Python 3.11.9 y bibliotecas que se necesitarán importar al principio del código como:
- [pandas](https://pandas.pydata.org/docs/)
- [numpy](https://numpy.org/doc/2.1/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [matplotlib-inline](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html)
- [seaborn](https://seaborn.pydata.org/)
- [requests](https://requests.readthedocs.io/en/latest/)
- [sys](https://docs.python.org/3/library/sys.html)
- [os](https://docs.python.org/3/library/os.html)
- [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- [sklearn.metrics.mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [sklearn.model_selection.LeaveOneOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)
- [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
- [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [tqdm](https://tqdm.github.io/)
- [warnings](https://docs.python.org/3/library/warnings.html)
- [pandas.options.display](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)


***Aportación al Usuario🤝:***

El doble fin de este proyecto incluye tanto el propio aprendizaje y formación como la intención de crear un modleo predictivo del precio de alquiler en Madrid que el usuario pueda usar.


***Próximos pasos:***

En un futuro, se recomienda ser ma´s exhaustivo y variado en el preprocesamiento y especificar más modelos para poder hacer un predicción más precisa.. La herramientas que más útiles pueden ser son el uso de otras formas de machine learning, inteligencia artificial u otras opciones.
