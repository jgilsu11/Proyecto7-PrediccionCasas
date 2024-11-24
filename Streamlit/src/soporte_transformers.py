import pickle

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('modelos/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('modelos/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('modelos/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return target_encoder, scaler, model
