import pickle

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('pickle_general/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    with open('pickle_general/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pickle_general/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return target_encoder, scaler, model
