import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import io

# ==========================================
# 1. Configuración Estética (El "Look & Feel")
# ==========================================
st.set_page_config(
    page_title="Art AI - Clasificador", 
    page_icon="🎨", 
    layout="centered"
)

# Estilos CSS personalizados para imitar el diseño
# - Fondo oscuro
# - Contenedor punteado para la imagen
# - Botones redondeados y personalizados
st.markdown("""
    <style>
    /* Fondo principal de la App */
    .stApp {
        background-color: #1a223a;
        color: #e0e0e0;
    }
    
    /* Centrar contenido principal */
    .block-container {
        padding-top: 2rem;
        max-width: 600px;
    }

    /* Subtítulo central */
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 0.9rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* Zona de previsualización de imagen punteada */
    .image-container {
        border: 2px dashed #404a6e;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background-color: rgba(255,255,255,0.03);
        margin-bottom: 20px;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    /* Texto de instrucción dentro de la zona punteada */
    .image-placeholder-text {
        color: #a0a0a0;
        font-size: 0.9rem;
    }

    /* Estilos para los botones (Cámara y Galería) */
    .stButton > button {
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        transition: all 0.2s ease;
        border: none;
    }

    /* Botón de Cámara (Morado) */
    div[data-testid="column"] button:first-child {
        background-color: #7d3c98;
        color: white;
    }
    div[data-testid="column"] button:first-child:hover {
        background-color: #8e44ad;
    }

    /* Botón de Galería (Azul) */
    div[data-testid="column"]:nth-child(2) button:first-child {
        background-color: #34495e;
        color: white;
    }
    div[data-testid="column"]:nth-child(2) button:first-child:hover {
        background-color: #405a74;
    }

    /* Botón de Analizar (Lila) */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #a569bd;
        color: white;
        margin-top: 15px;
        width: 100%;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #b97fcf;
    }

    /* Ocultar el uploader por defecto de Streamlit */
    div[data-testid="stFileUploader"] {
        display: none;
    }
    
    /* Ocultar la cámara por defecto de Streamlit */
    div[data-testid="stCameraInput"] {
        display: none;
    }

    </style>
""", unsafe_allow_html=True)

# Título Principal (模仿 imagen de referencia)
st.markdown('<h1 style="text-align: center; color: white;">🎨 Art AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Clasificador de Estilos Artísticos · 100% en dispositivo · sin internet</p>', unsafe_allow_html=True)


# ==========================================
# 2. Carga y Lógica del Modelo
# ==========================================
@st.cache_resource
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
    # Reemplaza con la ruta real de tu modelo
    model.load_state_dict(torch.load('best_model_ResNet18.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
class_names = ['Abstract Expressionism', 'Cubism', 'Expressionism', 'Impressionism', 'Symbolism']

definitions = {
    'Abstract Expressionism': 'Enfoque en formas y colores más que en objetos reales. ¡Pura emoción!',
    'Cubism': 'Uso de formas geométricas para ver objetos desde muchos ángulos a la vez. (Piensa en Picasso).',
    'Expressionism': 'Busca expresar sentimientos intensos, a menudo con colores irreales y formas distorsionadas.',
    'Impressionism': 'Captura la luz y el movimiento del momento con pinceladas cortas y visibles.',
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
    return prob

# ==========================================
# 3. Interfaz de Usuario (El Diseño Nuevo)
# ==========================================

# Zona de previsualización (el rectángulo punteado)
# -----------------------------------------------
image_placeholder = st.empty()

# Mensaje inicial si no hay imagen
# Usamos HTML para que tenga el estilo punteado
image_placeholder.markdown("""
    <div class="image-container">
        <p class="image-placeholder-text">Suba una obra o tome una foto para identificar su estilo.</p>
    </div>
""", unsafe_allow_html=True)

# Botones de entrada (imitando a la imagen de referencia)
# --------------------------------------------------------
col1, col2 = st.columns(2)

# Elementos invisibles para capturar la entrada
# (Están ocultos por CSS, se activan con código JavaScript/Streamlit)
uploaded_file_invisible = st.file_uploader("", type=["jpg", "jpeg", "png"], key="invisible_uploader")
camera_photo_invisible = st.camera_input("", key="invisible_camera")

img_source = None

# Botones visibles
# ---------------
with col1:
    # El botón "Cámara" activa la entrada de la cámara
    if st.button("📷 Cámara"):
        # JavaScript para hacer clic en el input de la cámara (este es el "truco")
        st.markdown("""
            <script>
                // Encontrar el input de la cámara y hacerle clic
                const cameraInput = window.parent.document.querySelectorAll('input[type="file"][accept*="image/*"]')[1];
                if (cameraInput) {
                    cameraInput.click();
                }
            </script>
        """, unsafe_allow_html=True)

with col2:
    # El botón "Galería" activa el uploader de archivos
    if st.button("🖼️ Galería"):
        # JavaScript para hacer clic en el uploader (este es el "truco")
        st.markdown("""
            <script>
                // Encontrar el uploader de archivos y hacerle clic
                const fileUploaderInput = window.parent.document.querySelector('input[type="file"][accept*="image/*"]');
                if (fileUploaderInput) {
                    fileUploaderInput.click();
                }
            </script>
        """, unsafe_allow_html=True)


# Lógica para mostrar la imagen seleccionada y habilitar el análisis
# ------------------------------------------------------------------
# Revisamos si hay entrada de la cámara o de la galería
image_to_analyze = None

if camera_photo_invisible:
    image_to_analyze = camera_photo_invisible
elif uploaded_file_invisible:
    image_to_analyze = uploaded_file_invisible

# Si hay una imagen, la mostramos en lugar del rectángulo punteado
if image_to_analyze:
    pil_image = Image.open(image_to_analyze).convert('RGB')
    
    # Reemplazamos el marcador punteado con la imagen real
    image_placeholder.markdown("""
        <div class="image-container">
            <p class="image-placeholder-text">Cargando...</p>
        </div>
    """, unsafe_allow_html=True) # Un pequeño estado de carga
    
    # Mostrar la imagen
    image_placeholder.image(pil_image, caption='Obra a analizar', use_container_width=True)
    
    # Habilitar el botón de analizar
    if st.button("⚡ Analizar estilo"):
        # -- Lógica de predicción --
        with st.spinner('Analizando...'):
            probs = predict(pil_image)
            top_prob, top_idx = torch.max(probs, 0)
            
            # -- Mostrar Resultados --
            st.divider()
            
            # Resultado principal
            st.subheader(f"Estilo principal: **{class_names[top_idx]}**")
            st.write(f"Confianza: **{top_prob:.2%}**")
            st.info(definitions.get(class_names[top_idx], "Estilo artístico fascinante."))

            # Otros estilos
            st.write("### Probabilidades por estilo:")
            # Creamos un DataFrame para la tabla y gráfico
            chart_data = pd.DataFrame({
                'Estilo': class_names,
                'Probabilidad': [float(p) for p in probs]
            }).sort_values(by='Probabilidad', ascending=False)
            
            # Gráfico de barras
            st.bar_chart(chart_data.set_index('Estilo'), use_container_width=True)