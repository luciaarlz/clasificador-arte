import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# 1. CONFIGURACIÓN ESTÉTICA "ART GALLERY"
# ==========================================
st.set_page_config(page_title="ArtVision AI", page_icon="🎭", layout="centered")

st.markdown("""
    <style>
    /* Fondo degradado estilo galería moderna */
    .stApp {
        background: radial-gradient(circle, #1e1e26 0%, #111116 100%);
        color: #f0f0f0;
    }
    
    /* Contenedor del "Marco" del cuadro */
    .art-frame {
        border: 1px solid #3d3d4d;
        border-radius: 20px;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.02);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 25px;
    }

    /* Títulos elegantes en serif */
    h1 {
        font-family: 'serif';
        font-weight: 700;
        letter-spacing: 3px;
        background: -webkit-linear-gradient(#ffffff, #777);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }

    /* Estilo para los botones */
    .stButton>button {
        border-radius: 12px;
        border: 1px solid #4a4a5a;
        background-color: transparent;
        color: #e0e0e0;
        width: 100%;
        height: 3.5em;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 13px;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
        border-color: #ffffff;
        transform: translateY(-2px);
    }

    /* Botón de Analizar Destacado */
    .analyze-btn button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        margin-top: 20px;
    }

    /* Ajustes para inputs nativos */
    .stCameraInput { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LÓGICA DEL MODELO (PYTORCH)
# ==========================================
@st.cache_resource
def load_model():
    # Definir arquitectura idéntica al entrenamiento
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 5), # 5 clases artísticas
    )
    # Cargar pesos (asegúrate de que el archivo esté en GitHub)
    try:
        model.load_state_dict(torch.load('best_model_ResNet18.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'best_model_ResNet18.pth'. Por favor súbelo a GitHub.")
    model.eval()
    return model

model = load_model()
class_names = ['Abstract Expressionism', 'Cubism', 'Expressionism', 'Impressionism', 'Symbolism']

definitions = {
    'Abstract Expressionism': 'Enfoque en formas y colores más que en objetos reales. ¡Pura emoción!',
    'Cubism': 'Uso de formas geométricas para ver objetos desde muchos ángulos a la vez (Estilo Picasso).',
    'Expressionism': 'Busca expresar sentimientos intensos, a menudo con colores irreales y formas distorsionadas.',
    'Impressionism': 'Captura la luz y el movimiento del momento con pinceladas cortas y visibles (Estilo Monet).',
    'Symbolism': 'Uso de imágenes metafóricas para representar ideas, sueños o estados de ánimo.'
}

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
        confidence, index = torch.max(prob, 0)
    return class_names[index], confidence.item(), prob

# ==========================================
# 3. INTERFAZ DE USUARIO
# ==========================================
st.markdown("<h1>ARTVISION AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; margin-bottom: 30px;'>Curador Digital de Estilos Artísticos</p>", unsafe_allow_html=True)

# Marcador de posición para el marco del cuadro
image_placeholder = st.empty()

# Si no hay imagen, mostramos el marco vacío con diseño artístico
image_placeholder.markdown("""
    <div class="art-frame">
        <div style="font-size: 60px; margin-bottom: 20px;">🖼️</div>
        <p style="color: #888; font-style: italic; font-size: 1.1rem;">"El arte no reproduce lo visible, sino que lo hace visible"</p>
        <p style="font-size: 0.8rem; color: #555; text-transform: uppercase; letter-spacing: 1px;">Sube una obra o captura una foto</p>
    </div>
""", unsafe_allow_html=True)

# Botones de selección
col1, col2 = st.columns(2)
with col1:
    img_file = st.file_uploader("🖼️ GALERÍA", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    st.markdown("<p style='text-align:center; font-size:12px; color:#666;'>GALERÍA</p>", unsafe_allow_html=True)

with col2:
    cam_file = st.camera_input("📷 CÁMARA", label_visibility="collapsed")
    st.markdown("<p style='text-align:center; font-size:12px; color:#666;'>CÁMARA</p>", unsafe_allow_html=True)

# Lógica principal de ejecución
final_img = img_file if img_file else cam_file

if final_img:
    image = Image.open(final_img).convert('RGB')
    # Colocamos la imagen en el marco (usando width='stretch' para 2026)
    image_placeholder.image(image, width='stretch')
    
    # Botón de análisis
    st.markdown("<div class='analyze-btn'>", unsafe_allow_html=True)
    if st.button("✨ IDENTIFICAR ESTILO"):
        with st.spinner('Analizando pinceladas...'):
            label, acc, all_probs = predict(image)
            
            st.divider()
            
            # Resultado Principal
            st.markdown(f"### Estilo detectado: <span style='color:#a569bd'>{label}</span>", unsafe_allow_html=True)
            st.write(f"**Confianza:** {acc:.2%}")
            st.info(definitions.get(label, "Un estilo fascinante."))

            # Gráfico de todas las probabilidades
            st.write("#### Probabilidades por categoría:")
            chart_data = pd.DataFrame({
                'Estilo': class_names,
                'Probabilidad': [float(p) for p in all_probs]
            }).sort_values(by='Probabilidad', ascending=False)
            
            st.bar_chart(chart_data.set_index('Estilo'), width='stretch')
            
    st.markdown("</div>", unsafe_allow_html=True)