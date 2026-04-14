import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# 1. Configuración estética
st.set_page_config(page_title="Clasificador de Arte", page_icon="🎨", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Clasificador de Arte")
st.write("Identifica el estilo artístico de cualquier obra con Inteligencia Artificial.")

# 2. Carga del Modelo
@st.cache_resource # Esto hace que el modelo se cargue una sola vez y no cada vez que tocas un botón
def load_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 5),
    )
    model.load_state_dict(torch.load('best_model_ResNet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ['Abstract Expressionism', 'Cubism', 'Expressionism', 'Impressionism', 'Symbolism']

# 3. Lógica de Predicción
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
    return prob

# 4. Interfaz de Usuario
col1, col2 = st.columns([1, 1])

with col1:
    option = st.radio("Método de entrada:", ("📤 Subir archivo", "📷 Usar cámara"))
    if option == "📤 Subir archivo":
        img_file = st.file_uploader("Imagen...", type=["jpg", "jpeg", "png"])
    else:
        img_file = st.camera_input("Capturar")

with col2:
    if img_file:
        image = Image.open(img_file).convert('RGB')
        st.image(image, caption='Obra a analizar', width='stretch')
        btn = st.button("ANALIZAR ESTILO")

if img_file and 'btn' in locals() and btn:
    probs = predict(image)
    top_prob, top_idx = torch.max(probs, 0)
    
    st.divider()
    
    # Resultado principal
    st.subheader(f"Resultado: **{class_names[top_idx]}**")
    st.progress(float(top_prob))
    st.write(f"Confianza del {top_prob:.2%}")

    # Otros estilos
    st.write("### Probabilidades por estilo:")
    # Creamos un DataFrame para la tabla y gráfico
    chart_data = pd.DataFrame({
        'Estilo': class_names,
        'Probabilidad': [float(p) for p in probs]
    }).sort_values(by='Probabilidad', ascending=False)
    
    st.bar_chart(chart_data.set_index('Estilo'))