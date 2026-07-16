from utils.embedding_selector import EmbeddingSelector
import numpy as np

# Initialize parameters
used_indexes = []  # Previously used embedding indexes
new_embed = np.random.randn(704)  # Your new embedding vector
ecapa = True  # Set to True for ECAPA vectors, False for full embeddings

# Select model path based on embedding type
if ecapa:
    new_embed = new_embed[-192:]  # Use last 192 dimensions for ECAPA
    model_path = 'model/ecapa/NVAE_12'
else:
    model_path = 'model/full_embedding/NVAE_12'

# # Initialize selector and load clustering model
selector = EmbeddingSelector(clusterer_path=model_path)

# # Select dissimilar embeddings
# selected_indices = selector.select_embeddings(
#     new_embedding=new_embed,
#     used_indexes=used_indexes,  # This list gets modified
#     n_select=8  # Number of embeddings to select
# )

# print(f"Selected embedding indices: {selected_indices}")

# def both_selections(new_embed):
    
    

for i in range(5000):
    new_embed = np.random.randn(704)[-192:]  # Generate a new random embedding vector

    try: 
        # selector = EmbeddingSelector(clusterer_path=model_path)

        # Select dissimilar embeddings
        selected_indices = selector.select_embeddings(
            new_embedding=new_embed,
            used_indexes=used_indexes,  # This list gets modified
        )
        used_indexes.extend(selected_indices)
        print(f"Iteration {i}, Number of used indexes: {len(used_indexes)}")
    except Exception as e:
        print(f"Error during selection: {e} at iteration {i+1}")
        raise e
    

    # print(f"Iteration {i+1}: Selected embedding indices: {selected_indices}")