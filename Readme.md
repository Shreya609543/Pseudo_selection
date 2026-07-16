# NVAE Clustering and Embedding Selection

A Python toolkit for clustering analysis and embedding selection using NVAE embeddings. This repository provides utilities for performing clustering analysis on embeddings and selecting dissimilar embeddings based on clustering results.

## Features

- **Embedding Selection**: Select dissimilar embeddings from existing clusters
- **Model Support**: Support for both ECAPA and full embeddings
- **Training** : 
    - **Clustering Analysis**: Complete clustering analysis with multiple evaluation metrics
    - **Save/Load Functionality**: Persistent storage of clustering models
    - **Visualization**: Built-in plotting and analysis tools

## Installation


1. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

## File Structure

```
├── example.py                          # Main usage example (embedding selection)
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── utils/                             # Utility modules
│   ├── embedding_selector.py          # Embedding selection utilities (PRIMARY)
│   ├── clustering.py                  # Clustering analysis tools (reference)
│   └── save_load.py                   # Model save/load functions (supporting)
└── model/                             # Pre-trained clustering models (PRIMARY)
    ├── ecapa/NVAE_12                  # ECAPA embedding models
    └── full_embedding/NVAE_12         # Full embedding models
```

## Usage

#### Embedding Selection

```python
from utils.embedding_selector import EmbeddingSelector

# Initialize selector and load clustering model
selector = EmbeddingSelector(clusterer_path=model_path)

# Select dissimilar embeddings
selected_indices = selector.select_embeddings(
    new_embedding=new_embed,
    used_indexes=used_indexes,  # This list gets modified
    n_select=8  # Number of embeddings to select
)

print(f"Selected embedding indices: {selected_indices}")
```


#### Clustering 

```python
from utils.clustering import run_complete_clustering_analysis

our_dir = ""
# Run complete clustering analysis
clusterer = run_complete_clustering_analysis(
    npz_file_path, 
    max_k=8, 
    max_samples=6000,
    out_path= out_dir,
    selection_strategy='max'  # Options: 'max', 'min', 'consensus', 'silhouette', 'elbow', 'calinski', 'davies'
)
```

#### Save and Load Clustering Models

```python
from utils.save_load import save_clusterer_components, load_clusterer_components

# Save clustering model
save_clusterer_components(clusterer, 'output/my_clusterer')

# Load clustering model
loaded_clusterer = load_clusterer_components('output/my_clusterer')
print(f"Loaded: {loaded_clusterer.n_embeddings} embeddings, {loaded_clusterer.optimal_k} clusters")
```

## Model Information

### ECAPA Models
- Located in `model/ecapa/NVAE_12`
- Uses the last 192 dimensions of input embeddings

### Full Embedding Models  
- Located in `model/full_embedding/NVAE_12`
- Uses complete embedding vectors (704 dimensions)
- General purpose embedding clustering


