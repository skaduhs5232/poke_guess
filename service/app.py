import os
import json
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# Configurações
BASE_DIR = Path(__file__).parent.parent
IMG_SIZE = 224
MODEL_PATH = BASE_DIR / "model" / "pokemon_sketch_classifier.keras"
LABEL_MAP_PATH = BASE_DIR / "model" / "label_map.json"
RAW_IMAGES_DIR = BASE_DIR / "notebooks" / "dataset" / "raw"

# Cache de imagens de Pokémon da PokeAPI
POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon/"


@st.cache_resource
def load_model():
    """Carrega o modelo de classificação"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}\n"
            "Execute o notebook 'pokemon_sketch_classifier.ipynb' primeiro para treinar o modelo."
        )
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return model


@st.cache_data
def load_label_map():
    """Carrega o mapeamento de labels"""
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Label map não encontrado em {LABEL_MAP_PATH}\n"
            "Execute o notebook 'pokemon_sketch_classifier.ipynb' primeiro para treinar o modelo."
        )
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    return label_map


def preprocess_image(img):
    """Preprocessa imagem para o modelo (MobileNetV2 format)"""
    # Converte PIL Image para numpy array
    img_array = np.array(img)
    
    # Se for RGBA, converte para grayscale considerando alpha
    if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
        alpha = img_array[:, :, 3:4] / 255.0
        rgb = img_array[:, :, :3]
        white_bg = np.ones_like(rgb) * 255
        img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    
    # Converter para grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Redimensionar
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Converter de volta para RGB (3 canais)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # Normalizar para [-1, 1] (padrão MobileNetV2)
    img_normalized = img_rgb.astype(np.float32) / 127.5 - 1.0
    
    # Adiciona dimensão de batch
    return np.expand_dims(img_normalized, axis=0)


def get_pokemon_image(pokemon_name):
    """Obtém a imagem do Pokémon (local ou da API)"""
    # Tenta carregar imagem local primeiro
    local_path = os.path.join(RAW_IMAGES_DIR, pokemon_name, "sprite.png")
    if os.path.exists(local_path):
        return Image.open(local_path)
    
    # Se não encontrar, busca da PokeAPI
    try:
        response = requests.get(f"{POKEAPI_URL}{pokemon_name}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            img_url = data["sprites"]["other"]["official-artwork"]["front_default"]
            if img_url:
                img_response = requests.get(img_url, timeout=5)
                return Image.open(BytesIO(img_response.content))
    except:
        pass
    
    return None


def predict_pokemon(image, model, label_map, top_k=5):
    """Faz a predição dos Pokémon mais similares"""
    # Preprocessa a imagem
    processed = preprocess_image(image)
    
    # Faz predição
    predictions = model.predict(processed, verbose=0)[0]
    
    # Obtém top-k índices
    top_idx = np.argsort(predictions)[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_idx):
        pokemon_name = label_map['idx_to_name'][str(idx)]
        confidence = float(predictions[idx])
        results.append({
            "pokemon": pokemon_name,
            "confidence": confidence,
            "rank": i + 1
        })
    
    return results


def main():
    # Configuração da página
    st.set_page_config(
        page_title="PokéGuess",
        page_icon="🎨",
        layout="wide"
    )
    
    # CSS customizado
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #FFCB05;
            text-shadow: 2px 2px 4px #3D7DCA;
            font-size: 3em;
            margin-bottom: 0;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            color: #666;
            margin-top: 0;
        }
        .pokemon-name {
            font-size: 2em;
            font-weight: bold;
            text-transform: capitalize;
            color: #3D7DCA;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #FFCB05 0%, #FF6B6B 100%);
            color: white;
            font-weight: bold;
            font-size: 1.2em;
            border: none;
            padding: 15px;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #FF6B6B 0%, #FFCB05 100%);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Título
    st.markdown("<h1 class='main-title'>🎨 PokéGuess</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Desenhe um Pokémon da Geração 1 ou 2 e descubra qual é!</p>", unsafe_allow_html=True)
    
    # Carrega modelo e label map
    with st.spinner("Carregando modelo..."):
        try:
            model = load_model()
            label_map = load_label_map()
            st.success(f"✅ Modelo carregado com sucesso! ({label_map['num_classes']} Pokémon)")
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.info("📝 **Como treinar o modelo:**\n\n"
                   "1. Abra o notebook `notebooks/pokemon_sketch_classifier.ipynb`\n"
                   "2. Execute todas as células\n"
                   "3. Aguarde o treinamento (pode levar alguns minutos)\n"
                   "4. Volte aqui e recarregue a página")
            return
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {e}")
            st.info(f"Caminho do modelo: {MODEL_PATH}")
            return
    
    # Inicializar session state para canvas
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("✏️ Desenhe aqui")
        
        # Configurações do canvas na sidebar ou inline
        stroke_width = st.slider("Espessura do traço", 1, 25, 8)
        stroke_color = st.color_picker("Cor do traço", "#000000")
        
        # Canvas para desenhar
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # Transparente
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )
        
        # Botões
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            top_k = st.slider("Quantidade de resultados", 3, 10, 5)
        
        with col_btn2:
            st.write("")
            st.write("")
            if st.button("🗑️ Limpar", use_container_width=True):
                st.session_state.canvas_key += 1
                if 'results' in st.session_state:
                    del st.session_state['results']
                st.rerun()
        
        # Botão de predição
        if st.button("🔍 Identificar Pokémon!", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                img_data = canvas_result.image_data
                # Verifica se há algo desenhado (não é todo branco)
                if np.any(img_data[:, :, :3] != 255):
                    with st.spinner("🔮 Analisando seu desenho..."):
                        image = Image.fromarray(img_data.astype("uint8"), "RGBA")
                        results = predict_pokemon(image, model, label_map, top_k)
                        st.session_state["results"] = results
                else:
                    st.warning("⚠️ Desenhe algo primeiro!")
            else:
                st.warning("⚠️ Desenhe algo primeiro!")
    
    with col2:
        st.subheader("🏆 Resultado")
        
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            
            # Pokémon principal (melhor match)
            winner = results[0]
            
            st.markdown("### 🥇 Melhor Match")
            
            winner_col1, winner_col2 = st.columns([1, 1.5])
            
            with winner_col1:
                winner_img = get_pokemon_image(winner["pokemon"])
                if winner_img:
                    st.image(winner_img, use_container_width=True)
                else:
                    st.info("🖼️ Imagem não disponível")
            
            with winner_col2:
                st.markdown(f"<p class='pokemon-name'>{winner['pokemon']}</p>", unsafe_allow_html=True)
                confidence_pct = winner["confidence"] * 100
                st.progress(min(winner["confidence"], 1.0))
                st.markdown(f"**Confiança:** {confidence_pct:.1f}%")
                
                st.markdown(f"""
                    <a href="https://pokemon.fandom.com/pt-br/wiki/{winner['pokemon']}" target="_blank">
                        📖 Ver na Pokédex
                    </a>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Outros Pokémon próximos
            if len(results) > 1:
                st.markdown("### 📊 Pokémon similares")
                
                num_cols = 4
                cols = st.columns(num_cols)
                
                for i, result in enumerate(results[1:]):
                    with cols[i % num_cols]:
                        pokemon_img = get_pokemon_image(result["pokemon"])
                        
                        st.markdown(f"**#{result['rank']} {result['pokemon'].capitalize()}**")
                        
                        if pokemon_img:
                            st.image(pokemon_img, use_container_width=True)
                        else:
                            st.info("🖼️")
                        
                        confidence_pct = result["confidence"] * 100
                        st.progress(min(result["confidence"], 1.0))
                        st.caption(f"{confidence_pct:.1f}%")
        else:
            st.info("👈 Desenhe um Pokémon no canvas e clique em 'Identificar'!")
            
            st.markdown("""
            ### 💡 Dicas para melhores resultados:
            
            - **Desenhe de forma clara** - Linhas bem definidas funcionam melhor
            - **Formas simples** - A silhueta é importante!
            - **Use toda a área** - Desenhe grande
            - **Características marcantes** - Orelhas do Pikachu, cauda do Charizard...
            
            ### 🎯 Pokémon disponíveis: 
            """)
            st.caption(f"O modelo conhece **{label_map['num_classes']} Pokémon** das **Gerações 1 e 2** (#001 Bulbasaur até #251 Celebi)!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>🎮 PokéGuess - Feito com ❤️ e TensorFlow | "
        "Desenvolvido por <a href='https://github.com/skaduhs5232' target='_blank' style='color: #3D7DCA;'>Thiago</a></p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
