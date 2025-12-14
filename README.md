# 🎨 PokéGuess - Classificador de Sketches de Pokémon

Sistema de reconhecimento de desenhos de Pokémon usando Deep Learning com Transfer Learning. O modelo identifica qual Pokémon foi desenhado e retorna o nome com a confiança da predição.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Arquitetura Técnica](#arquitetura-técnica)
- [Dataset](#dataset)
- [Pipeline de Treinamento](#pipeline-de-treinamento)
- [Modelo](#modelo)
- [Data Augmentation](#data-augmentation)
- [Instalação e Uso](#instalação-e-uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Resultados](#resultados)
- [Tecnologias](#tecnologias)
- [Autor](#autor)

---

## 🎯 Visão Geral

O **PokéGuess** é um sistema de visão computacional que classifica desenhos (sketches) de Pokémon das gerações 1 e 2 (151 + 100 = 251 classes). O usuário desenha um Pokémon em uma interface web e o modelo retorna:

- Nome do Pokémon identificado
- Confiança da predição (%)
- Top-K pokémons mais similares

### Características principais:

- ✅ **Transfer Learning** com MobileNetV2 pré-treinado no ImageNet
- ✅ **251 classes** (Pokémon #001 Bulbasaur até #251 Celebi)
- ✅ **Data Augmentation** forte para compensar escassez de dados
- ✅ **Fine-tuning em 2 fases** para melhor convergência
- ✅ **Interface web interativa** com Streamlit
- ✅ **Preprocessing otimizado** para sketches monocromáticos

---

## 🏗️ Arquitetura Técnica

### Pipeline Completo

```
Desenho do Usuário (Canvas)
         ↓
  Preprocessing
    - RGBA → Grayscale
    - Resize para 224x224
    - Grayscale → RGB (3 canais)
    - Normalização [-1, 1]
         ↓
  MobileNetV2 (Feature Extractor)
    - Base congelada (inicialmente)
    - 1280 features extraídas
         ↓
  Classificador Custom
    - Dense(512) + BatchNorm + ReLU + Dropout(0.5)
    - Dense(256) + BatchNorm + ReLU + Dropout(0.4)
    - Dense(251, softmax)
         ↓
  Predição
    - Top-K resultados
    - Confidence scores
```

### Decisões Técnicas

#### Por que Transfer Learning?

- **Poucos dados**: ~21 imagens por classe (5.271 imagens totais)
- **MobileNetV2 já aprendeu features visuais** úteis do ImageNet
- **Convergência mais rápida** e melhor generalização
- **Regularização implícita** pela base pré-treinada

#### Por que MobileNetV2?

- **Leve e rápido**: 3.5M parâmetros (vs ResNet50 25M)
- **Bom para deployment**: Ideal para aplicações web
- **Depthwise Separable Convolutions**: Eficiência computacional
- **Excelente para imagens 224x224**: Tamanho nativo

---

## 📊 Dataset

### Composição

O dataset combina duas fontes:

1. **Synthetic Sketches** (pré-existentes)
   - Sketches sintéticos gerados de sprites
   - ~9 imagens por Pokémon
   
2. **PokeAPI Sketches** (gerados no notebook)
   - Sprites oficiais da PokeAPI
   - Convertidos para sketch usando 3 métodos:
     - **Canny Edge Detection**: Bordas nítidas
     - **Pencil Sketch**: Estilo lápis
     - **Laplacian**: Detecção de gradientes
   - 5 sprites × 3 métodos = 15 imagens por Pokémon

### Processamento de Imagens da PokeAPI

```python
def image_to_sketch(image, method='canny'):
    # Canny: Blur + Edge Detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    sketch = 255 - edges  # Inverter: linhas pretas em fundo branco
```

### Estatísticas do Dataset

- **Total de classes**: 251 Pokémon
- **Total de imagens**: ~5.271
- **Média por classe**: ~21 imagens
- **Split**: 80% treino / 10% validação / 10% teste
- **Formato**: PNG, fundo branco, linhas pretas

---

## 🔄 Pipeline de Treinamento

### Fase 1: Treinamento do Classificador (Base Congelada)

```python
# Congelar base do MobileNetV2
base_model.trainable = False

# Compilar
model.compile(
    optimizer=Adam(lr=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar (30 épocas)
history1 = model.fit(train_generator, ...)
```

**Objetivo**: Treinar o classificador custom sem alterar os pesos da base pré-treinada.

### Fase 2: Fine-Tuning (Últimas 30 Camadas Descongeladas)

```python
# Descongelar últimas 30 camadas
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompilar com LR muito menor
model.compile(
    optimizer=Adam(lr=1e-5),  # 100x menor!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar (50 épocas)
history2 = model.fit(train_generator, ...)
```

**Objetivo**: Ajustar finamente as últimas camadas da base para adaptar às características dos sketches.

### Callbacks Utilizados

```python
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        mode='max'
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

---

## 🤖 Modelo

### Arquitetura Detalhada

```python
Input (224, 224, 3)
    ↓
MobileNetV2 Base (weights='imagenet', pooling='avg')
    - 1280 features
    ↓
Dense(512) + L2(0.01)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
Dropout(0.5)
    ↓
Dense(256) + L2(0.01)
    ↓
BatchNormalization
    ↓
ReLU
    ↓
Dropout(0.4)
    ↓
Dense(251, activation='softmax')
```

### Parâmetros do Modelo

- **Total de parâmetros**: ~3.7M
- **Treináveis (Fase 1)**: ~850K (apenas classificador)
- **Treináveis (Fase 2)**: ~1.5M (classificador + últimas 30 camadas)
- **Congelados**: ~2.2M (maior parte do MobileNetV2)

### Regularização

- **L2 Regularization** (0.01) nas camadas Dense
- **Dropout** (0.5 e 0.4) para prevenir overfitting
- **BatchNormalization** para estabilizar treinamento
- **Data Augmentation** (ver próxima seção)

---

## 🔀 Data Augmentation

Para compensar a **escassez de dados** (~21 imagens/classe), aplicamos data augmentation **forte**:

```python
ImageDataGenerator(
    rotation_range=30,           # Rotação ±30°
    width_shift_range=0.2,       # Shift horizontal 20%
    height_shift_range=0.2,      # Shift vertical 20%
    shear_range=0.2,             # Shear 20%
    zoom_range=0.2,              # Zoom ±20%
    horizontal_flip=True,        # Flip horizontal
    vertical_flip=False,         # Não flip vertical
    brightness_range=[0.8, 1.2], # Variação de brilho
    fill_mode='constant',
    cval=1.0                     # Preencher com branco
)
```

### Justificativa

- **Rotação**: Sketches podem ser desenhados em qualquer ângulo
- **Shift e Zoom**: Simula diferentes tamanhos e posições
- **Shear**: Adiciona variação geométrica
- **Brilho**: Compensa diferentes intensidades de traço
- **No vertical flip**: Pokémon têm orientação definida

---

## 🚀 Instalação e Uso

### Pré-requisitos

- Python 3.11+
- pip

### 1. Clonar o Repositório

```bash
git clone https://github.com/skaduhs5232/poke_guess.git
cd poke_guess
```

### 2. Instalar Dependências

```bash
pip install -r service/requirements.txt
```

### 3. Treinar o Modelo (Opcional)

Se quiser retreinar o modelo:

```bash
# Abrir o notebook
jupyter notebook notebooks/pokemon_sketch_classifier.ipynb

# Executar todas as células (Ctrl+A, Shift+Enter)
# Aguardar o treinamento (~30-60 min dependendo do hardware)
```

### 4. Executar a Aplicação Web

```bash
streamlit run service/app.py
```

A aplicação abrirá em `http://localhost:8501`

### 5. Usar o Modelo

1. Desenhe um Pokémon no canvas
2. Clique em "🔍 Identificar Pokémon!"
3. Veja o resultado com confiança

---

## 📁 Estrutura do Projeto

```
poke_guess/
│
├── notebooks/
│   ├── pokemon_sketch_classifier.ipynb  # Notebook principal de treinamento
│   └── dataset/
│       ├── synthetic_sketches/          # Sketches sintéticos (entrada)
│       ├── pokeapi_sketches/            # Sketches gerados da PokeAPI
│       └── combined/                    # Dataset combinado final
│
├── model/
│   ├── pokemon_sketch_classifier.keras  # Modelo treinado
│   ├── best_model.keras                 # Melhor checkpoint
│   ├── label_map.json                   # Mapeamento idx ↔ nome
│   ├── metrics.json                     # Métricas de avaliação
│   └── training_history.png             # Gráfico de treinamento
│
├── service/
│   ├── app.py                           # Aplicação Streamlit
│   └── requirements.txt                 # Dependências
│
└── README.md                            # Este arquivo
```

---

## 📈 Resultados

### Métricas de Avaliação

As métricas exatas estão em `model/metrics.json`. Valores esperados:

- **Test Accuracy**: Varia conforme treinamento
- **Top-3 Accuracy**: Geralmente 30-50% maior que Top-1
- **Top-5 Accuracy**: Geralmente 50-70% maior que Top-1

### Considerações sobre Performance

Com **251 classes** e **~21 imagens por classe**, o modelo enfrenta:

- ✅ **Transfer Learning mitiga overfitting**
- ✅ **Data Augmentation aumenta diversidade**
- ⚠️ **Poucos dados ainda é um desafio**
- ⚠️ **Pokémon similares podem confundir** (ex: evoluções)

### Melhorias Futuras

Para melhorar o modelo:

1. **Mais dados**: Coletar sketches reais de usuários
2. **Few-Shot Learning**: Técnicas para classes com poucos exemplos
3. **Ensemble**: Combinar múltiplos modelos
4. **Contrastive Learning**: SimCLR, MoCo para melhor embedding
5. **Arquiteturas alternativas**: EfficientNet, Vision Transformer

---

## 🛠️ Tecnologias

### Machine Learning
- **TensorFlow 2.x**: Framework principal
- **Keras**: API de alto nível
- **MobileNetV2**: Arquitetura base (Transfer Learning)
- **NumPy**: Computação numérica
- **OpenCV**: Processamento de imagens
- **scikit-learn**: Split de dados

### Web Interface
- **Streamlit**: Framework web interativo
- **streamlit-drawable-canvas**: Canvas de desenho
- **Pillow**: Manipulação de imagens
- **Requests**: Consultas à PokeAPI

### Data Processing
- **Pandas**: (se usado para análise)
- **Matplotlib**: Visualização de gráficos
- **PokeAPI**: Fonte de sprites oficiais

---

## 👨‍💻 Autor

**Thiago**

[![GitHub](https://img.shields.io/badge/GitHub-skaduhs5232-181717?style=flat&logo=github)](https://github.com/skaduhs5232)

---

## 📄 Licença

Este projeto é open source e está disponível sob a licença MIT.

---

## 🙏 Agradecimentos

- **PokeAPI**: Por fornecer sprites oficiais dos Pokémon
- **TensorFlow/Keras**: Framework de Deep Learning
- **Streamlit**: Framework web rápido e intuitivo
- **Comunidade Pokémon**: Por inspirar este projeto

---

## 📚 Referências

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [PokeAPI Documentation](https://pokeapi.co/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Feito com ❤️ e TensorFlow**
