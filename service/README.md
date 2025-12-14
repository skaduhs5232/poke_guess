# PokéGuess Service

Aplicativo Streamlit para identificar Pokémon a partir de desenhos.

## Como Executar

### 1. Instalar dependências

```bash
cd service
pip install -r requirements.txt
```

### 2. Verificar arquivos do modelo

Certifique-se de que os seguintes arquivos existem:
- `model/pokemon_sketch_embedding_v2.keras` - Modelo treinado
- `model/pokemon_embeddings.npy` - Embeddings dos Pokémon
- `model/pokemon_labels.npy` - Labels dos Pokémon

### 3. Executar o app

```bash
streamlit run app.py
```

O app abrirá automaticamente no navegador em `http://localhost:8501`

## Funcionalidades

- 🔍 Identificação do Pokémon com confiança
- 🏆 Top-K Pokémon mais similares
- 🖼️ Imagens oficiais dos Pokémon
- 📖 Links para a Pokédex

## Estrutura de Arquivos

```
service/
├── app.py              # Aplicativo Streamlit
├── requirements.txt    # Dependências
└── README.md          # Este arquivo

model/
├── pokemon_sketch_embedding_v2.keras   # Modelo
├── pokemon_embeddings.npy              # Embeddings
└── pokemon_labels.npy                  # Labels
```

## Screenshots

### Tela Principal
Upload de desenho e visualização dos resultados

### Resultados
- Pokémon identificado com maior confiança
- Lista dos próximos Pokémon mais similares
- Barras de progresso mostrando confiança
