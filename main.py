import streamlit as st
import pandas as pd
import pickle
import numpy as np
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Predicción de Alquiler de Casas",
    page_icon="🏠",
    layout="centered",
)

# Título y descripción
st.title("🏠 Predicción de Alquiler de Casas")
st.write("Usa esta aplicación para predecir el precio de alquiler de una casa en Madrid basándote en sus características.")

# Mostrar una imagen
st.image(
    "C:\\Users\\jaime\\Desktop\\proyectos\\Proyecto 7\\Proyecto7-PrediccionCasas\\Imagen\\imagen alquiler.webp",  # URL de la imagen
    caption="Tu próximo alquiler está aquí.",
    use_column_width=True,
)


# Cargar los modelos y transformadores entrenados
def load_models():
    with open('transformers/one_hot_encoder.pkl', 'rb') as f:
        one_hot = pickle.load(f)    
    with open('transformers/target_encoder.pkl', 'rb') as t:
        target_encoder = pickle.load(t)
    with open('transformers/scaler.pkl', 'rb') as s:
        scaler = pickle.load(s)
    with open('transformers/random_forest_model.pkl', 'rb') as r:
        model = pickle.load(r)
    return one_hot,target_encoder, scaler, model

one_hot,target_encoder, scaler, model = load_models()

st.header("🔧 Características de la vivienda")
col1, col2 = st.columns(2)

with col1:
    propertyType = st.selectbox("Tipo de vivienda", ["flat", "penthouse", "studio", "duplex", 'chalet', 'countryHouse'], help="Selecciona el tipo de propiedad de la casa.")
    size = st.slider('Tamaño del Piso', 20,149,50, help='Elige el tamaño ideal para ti') #min,max,predeterminado
    exterior = st.selectbox("Exterior", ["True", "False"], help="Elige si tiene vista a la calle.")
    rooms = st.number_input("Habitaciones", min_value=0,max_value=4 , value=1, step=1, help="Elige un número de habitaciones entre 0 y 4.")
    bathrooms = st.number_input("Baños", min_value=1, max_value=3, value=1, step=1, help="Elige cuántos baños tiene.")
    province= st.selectbox("Provincia", ["Madrid", "Afueras"])
    municipality= st.selectbox("Municipio", ['Madrid', 'San Sebastián de los Reyes', 'Villamanrique de Tajo',
       'Recas', 'Cedillo del Condado', 'Rascafría', 'Manzanares el Real',
       'Miraflores de la Sierra', 'El Viso de San Juan', 'Galapagar',
       'Arganda', 'San Lorenzo de el Escorial', 'Camarena', 'Aranjuez',
       'Villanueva del Pardillo', 'Azuqueca de Henares', 'El Espinar',
       'Las Rozas de Madrid', 'Guadalajara', 'Illescas', 'Navalcarnero',
       'Seseña', 'Casarrubios del Monte', 'Alcalá de Henares',
       'El Escorial', 'Calypo Fado', 'Leganés', 'Coslada',
       'Torrejón de Ardoz', 'Marchamalo', 'Camarma de Esteruelas',
       'Alcorcón', 'Pinto', 'Valdemoro', 'Collado Villalba', 'Getafe',
       'Paracuellos de Jarama', 'El Molar', 'Parla', 'Tres Cantos',
       'Yuncos', 'Esquivias', 'Quijorna', 'Valdemorillo', 'Yuncler',
       'Pedrezuela', 'Daganzo de Arriba', 'Yeles', 'Guadarrama', 'Ocaña',
       'Cobeña', 'El Álamo', 'Algete', 'El Casar', 'Rivas-Vaciamadrid',
       'Los Santos de la Humosa', 'San Fernando de Henares',
       'Aldea del Fresno', 'Fuenlabrada', 'Fuensalida', 'Mataelpino',
       'Villa del Prado', 'Los Molinos', 'Colmenar Viejo', 'Móstoles',
       'Borox', 'Navalafuente', 'Robledo de Chavela', 'Campo Real',
       'Villaviciosa de Odón', 'Mocejón', 'San Ildefonso o la Granja',
       'Alameda de la Sagra', 'Cabañas de la Sagra',
       'Las Navas del Marqués', 'Villaseca de la Sagra',
       'Pozuelo de Alarcón', 'Yebes', 'Bustarviejo', 'Collado Mediano',
       'Chinchón', 'Valmojado', 'Alovera', 'Colmenarejo', 'Loeches',
       'Sevilla la Nueva', 'Serranillos del Valle',
       'Las Ventas de Retamosa', 'Torrelaguna', 'Villalbilla',
       'Alcobendas'])

with col2:
    distance = st.slider("Distancia del Centro", 183,59919,1000,help="Elige tu distancia deseada del Centro.")
    status = st.selectbox('Condición del piso', ['good', 'desconocido', 'newdevelopment', 'renew'], help='Elije las condiciones de tu nuevo piso') #cambiar desconocido
    floor = st.selectbox("Piso", ["ss", "st", 'bj', 'en', 'desconocido', '1','2', '3', '4', '5', '6', '7', '8', '14'], help="Elige si tiene vista a la calle.")
    district= st.selectbox("Distrito",['Hortaleza', 'Centro Urbano', 'desconocido', 'Puente de Vallecas',
       'Ciudad Lineal', 'Casco Antiguo', 'Moncloa', 'Centro',
       'Centro - Casco Histórico', 'Retiro', 'Arganzuela', 'Latina',
       'Barrio de Salamanca', 'Bulevar - Plaza Castilla', 'La Estación',
       'Barajas', 'Las Matas- Peñascales',
       'San Roque-Concordia-Adoratrices', 'Chamberí', 'Villaverde',
       'La Dehesa - El Pinar', 'Seseña Nuevo', 'Reyes Católicos',
       'Chorrillo', 'Valdepelayo - Montepinos - Arroyo Culebro',
       'Valleaguado - La Cañada', 'Suroeste',
       'San Isidro - Los Almendros', 'San José - Buenos Aires',
       'Hospital', 'Parque de la Coruña - Las Suertes',
       'Valderas - Los Castillos', 'Getafe Centro', 'San Blas', 'Val',
       'Casco Urbano', 'Casco Histórico', 'Los Llanos - Valle Pardo',
       'Ensanche', 'Dehesa - El Soto', 'El Vallejo', 'Pintores-Ferial',
       'Carabanchel', 'Zona Estación- Centro', 'Tetuán', 'El Quiñón',
       'Constitución-El Balconcillo', 'Valdemorillo pueblo',
       'Señorío de Illescas', 'Nuevo Aranjuez-Ciudad de las Artes',
       'Vega de la Moraleja', 'Villa de Vallecas', 'Fuencarral',
       'Noroeste', 'Fuentebella-San Felix-El Leguario', 'Rivas Futura',
       'Reyes', 'Parque Roma - Coronas', 'Parque Europa - Los Pitufos',
       'Vicálvaro', 'La Alhóndiga', 'Villalba Estación', 'Usera',
       'Zona Estación', 'Sudeste Industrial', 'Juan de Austria',
       'Montserrat - Parque Empresarial', 'Zona Industrial', 'Espartales',
       'Parque Inlasa', 'Universidad', 'Las Américas',
       'San Crispín - La Estación Consorcio', 'Foso-Moreras',
       'Getafe norte', 'Parla Este', 'Villayuventus-Renfe', 'Carlos Ruiz',
       'El Espinar', 'Chamartín', 'El Nido-Las Fuentes',
       'El Mirador - Grillero', 'La Espinilla - Parque Blanco',
       'Zona Pueblo', 'Los Ángeles de San Rafael', 'Ciudad 70',
       'Buenavista', 'Las Sedas - El Olivar', 'Las Cañas',
       'Las Lomas-Salinera-La Muñeca', 'El Mirador',
       'Pol. Industrial sur', 'Parque - Ctra de Ugena', 'San Isidro',
       'Pryconsa - Poligono Europa', 'Alcobendas Centro','desconocido'])
    hasLift = st.selectbox('Tiene ascensor', ['True', 'False', 'desconocido'])
    parkingSpace= st.selectbox('Tiene plaza de garaje', ['True', 'False', 'desconocido'])




# Botón para realizar la predicción
if st.button("💡 Predecir Precio"):
    # Crear DataFrame con los datos ingresados    #Comprobar que sean las mismas que la mía y el mismo orden
    new_house = pd.DataFrame({
        'propertyType': [propertyType],
        'size': [size],
        'exterior': [str(exterior)],
        'rooms': [str(rooms)],
        'bathrooms': [str(bathrooms)],
        'province' : [str(province)],
        'municipality' : [str(municipality)],
        'distance': [int(distance)],
        'status': [str(status)],
        'floor': [str(floor)],
        'district': [str(district)],
        'hasLift': [str(hasLift)],
        'parkingSpace': [str(parkingSpace)]
    })
  
    new_house=pd.DataFrame(new_house)
    
    col_encode = ["propertyType", "exterior", "rooms", "status", "floor", "hasLift", "parkingSpace"]
    onehot = one_hot.transform(new_house[col_encode])
    # Obtenemos los nombres de las columnas del codificador
    column_names = one_hot.get_feature_names_out(col_encode)
    # Convertimos a un DataFrame
    onehot_df = pd.DataFrame(onehot.toarray(), columns=column_names)
    #onehot_df.drop(columns=col_encode,inplace=True)                    
    # Columnas categóricas y numéricas                    #Poner aquí mis columnas concretas
    # categorical_columns = ['propertyType', 'exterior', 'rooms', 'bathrooms', 'status', 'floor', 'hasLift']
    numerical_columns = ['price', 'size', 'bathrooms', 'province', 'municipality', 'distance',
             'district', 'propertyType_chalet', 'propertyType_countryHouse',
             'propertyType_duplex', 'propertyType_flat', 'propertyType_penthouse',
             'propertyType_studio', 'exterior_False', 'exterior_True', 'rooms_0',
             'rooms_1', 'rooms_2', 'rooms_3', 'rooms_4', 'status_desconocido',
             'status_good', 'status_newdevelopment', 'status_renew', 'floor_1',
             'floor_14', 'floor_2', 'floor_3', 'floor_4', 'floor_5', 'floor_6',
             'floor_7', 'floor_8', 'floor_bj', 'floor_desconocido', 'floor_en',
             'floor_ss', 'floor_st', 'hasLift_False', 'hasLift_True',
             'hasLift_desconocido', 'parkingSpace_False', 'parkingSpace_True',
             'parkingSpace_desconocido']
   
    
    # # Aplicar el OneHotEncoder, TargetEncoder y StandardScaler
    # diccionario_encoding = {"onehot": ["propertyType", "exterior", "rooms", "status", "floor", "hasLift", "parkingSpace"], 
    #                     "dummies": [], # no metemos ninguna
    #                     'ordinal' : {}, #no metemos ninguna
    #                     "label": [] , # no metemos ninguna columna porque no queremos en ningún caso que se asignen las categorías de forma aleatoria
    #                     "frequency": [], # no metemos ninguna columna porque no coincide el orden del value counts con las categorias y la variable respuesta
    #                     "target": ["bathrooms", "province", "municipality", "district"]  
    #                     }
    # col_encode=diccionario_encoding.get("onehot", [])
    # st.write(col_encode)
    
             #Añadir el Onehot
    # Codificación de las columnas categóricas
    new_house.drop(columns = col_encode,inplace=True)
    new_house_encoded = pd.concat([new_house, onehot_df], axis=1)
    #new_house_encoded.drop(columns=col_encode,inplace=True)
    # new_house_encoded = pd.DataFrame()
    new_house_encoded["price"] = np.nan
    new_house_encoded = target_encoder.transform(new_house_encoded)
    

    new_house_encoded2=new_house_encoded.copy()
    new_house_encoded3=new_house_encoded.copy()
    # One-Hot Encoding
    # Hacemos el OneHot Encoder

    # st.write(onehot_df)

    # concatenamos los resultados obtenidos en la transformación con el DataFrame original
    #new_house.drop(columns=col_encode,inplace=True)



    # # Target Encoding (corrige el uso)
    # encoded_target = target_encoder.transform(new_house_encoded[["bathrooms", "province", "municipality", "district"]])

    # # Combina los datos codificados
    # new_house_encoded = pd.concat([new_house_encoded, encoded_target], axis=1)

    # Filtra las columnas numéricas y escala
    new_house_encoded2.drop(columns="price", inplace=True)
    new_house_encoded = scaler.transform(new_house_encoded2)
    new_house_encoded = pd.DataFrame(new_house_encoded)
    # new_house_encoded.drop(columns=6,inplace=True)
    new_house_encoded3["price"]=new_house_encoded[6]   #EL DATA FRAME AL PASAR POR EL SCALER NO PUEDE TENER PRICE PERO EN EL MODLEO SI QUE TIENE QUE ESTAR PRICE
    
    # Realizar la predicción
    prediction = model.predict(new_house_encoded)[0]
    # y_pred=modelo_final.predict(x)
    # Mostrar el resultado
    st.success(f"💵 El precio estimado del alquiler de la casa es: ${prediction}")
    st.balloons()

st.markdown(
    """
    ---
    **Proyecto creado con el potencial de la ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)
