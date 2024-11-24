import streamlit as st
import pandas as pd
import pickle

from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Predicción de Alquiler de Casas",
    page_icon="🏠",
    layout="centered",
)

# Título y descripción
st.title("🏠 Predicción de Alquiler de Casas con Machine Learning")
st.write("Usa esta aplicación para predecir el precio de alquiler de una casa en Madrid basándote en sus características.")

# Mostrar una imagen llamativa
st.image(
    "https://images.unsplash.com/photo-1568605114967-8130f3a36994",  # URL de la imagen
    caption="Tu próximo alquiler está aquí.",
    use_column_width=True,
)

#HASTA AQUÍ ESTÁ REVISADO

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('modelos/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('modelos/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('modelos/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return target_encoder, scaler, model

target_encoder, scaler, model = load_models()

# Formularios de entrada
st.header("🔧 Características de la casa")
col1, col2 = st.columns(2)

with col1:
    neighborhood = st.selectbox("Barrio", ["A", "B", "C", "D"], help="Selecciona el barrio de la casa.")
    house_type = st.selectbox("Tipo de Casa", ["Detached", "Semi-Detached"], help="Elige el tipo de casa.")

with col2:
    rooms = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3, step=1)
    area = st.number_input("Área en m²", min_value=50, max_value=500, value=120, step=10)

# Botón para realizar la predicción
if st.button("💡 Predecir Precio"):
    # Crear DataFrame con los datos ingresados
    new_house = pd.DataFrame({
        'Neighborhood': [neighborhood],
        'HouseType': [house_type],
        'Rooms': [rooms],
        'Area': [area],
    })

    # Columnas categóricas y numéricas
    categorical_columns = ['Neighborhood', 'HouseType']
    numerical_columns = ['Rooms', 'Area']


    #APLICAR LAS LIMPIEZAS CORRESPONDIENTES


    # Aplicar el TargetEncoder y StandardScaler
    new_house_encoded = new_house.copy()
    new_house_encoded[new_house.columns] = target_encoder.transform(new_house)
    new_house_encoded[numerical_columns] = scaler.transform(new_house_encoded[numerical_columns])

    # Realizar la predicción
    prediction = model.predict(new_house_encoded)[0]

    # Mostrar el resultado
    st.success(f"💵 El precio estimado de la casa es: ${prediction}")
    st.balloons()

# Pie de página
st.markdown(
    """
    ---
    **Proyecto creado con el potencial de la ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)