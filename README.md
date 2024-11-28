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
        ├── Imagen/                                      # Imagen para su uso en streamlit       
        ├── Modelos/                                     # Notebooks de Jupyter donde se han ido desarrollando los modelos
        ├── transformers/                                # Donde se han guardado los pickles para su uso en streamlit
        ├── Presentación/                                # Presentación de comparación de modelos        
        ├── src/                                         # Scripts (.py)
        ├── README.md                                    # Descripción del proyecto
        ├── main                                         # Script donde desarrollar streamlit (.py) 
        ├── base.yml/                                    # Entorno con todas las bibliotecas y versiones necesarias                  
        
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
- [sklearn](https://scikit-learn.org/stable/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [tqdm](https://tqdm.github.io/)
- [warnings](https://docs.python.org/3/library/warnings.html)
- [pandas.options.display](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html)
  
Además se dispone del archivo base.yml donde se puede clonar el entorno necesario para el funcionamiento del proyecto.     
***Tabla Resumen***  
  
| **Modelo**    | **Algoritmo**       | **R² (Train)** | **RMSE (Train)** | **R² (Test)** | **RMSE (Test)** | **Descripción**                                    |
|---------------|---------------------|----------------|------------------|---------------|-----------------|---------------------------------------------------|
| **Modelo 5*** | Random Forest       | 0.77           | 0.10             | -             | -               | Mejor modelo, entrenado con todos los datos y pickles. |
| **Modelo 3**  | Random Forest       | 0.75           | 41.90            | 0.76          | 40.12           | Buen desempeño en ambos conjuntos (train y test). |
| **Modelo 1**  | Random Forest       | 0.74           | 40.93            | 0.69          | 42.54           | Desempeño aceptable, pero menor al Modelo 3.     |
| **Modelo 2**  | Random Forest       | 0.41           | 39.29            | 0.33          | 42.50           | Desempeño significativamente más bajo.           |
| **Modelo 4**  | Regresión Lineal    | -              | -                | -             | -               | Peor modelo.                                      |

**Notas**:  
- *El Modelo 5 es una extensión del Modelo 3, entrenado con todos los datos y exportado con los pickles.  
- Las métricas del Modelo 4 no se incluyen debido a su pésimo desempeño.  
- Recordar que la base de datos cuenta únicamente con 400 filas por ello las métricas pueden ser más débiles  

***Aportación al Usuario🤝:***

El doble fin de este proyecto incluye tanto el propio aprendizaje y formación como la intención de crear un modleo predictivo del precio de alquiler en Madrid que el usuario pueda usar.


***Próximos pasos:***

En un futuro, se recomienda ser ma´s exhaustivo y variado en el preprocesamiento y especificar más modelos para poder hacer un predicción más precisa.. La herramientas que más útiles pueden ser son el uso de otras formas de machine learning, inteligencia artificial u otras opciones.
