import numpy as np
import joblib
import json
from pathlib import Path

def save_clusterer_components(clusterer, filepath='clusterer_data'):
    """Save clusterer components separately to avoid pickling errors."""
    
    # Create directory if it doesn't exist
    Path(filepath).mkdir(parents=True, exist_ok=True)
    
    # Save essential components
    components = {
        'embeddings': clusterer.embeddings_scaled,
        'embeddings_raw': getattr(clusterer, 'embeddings', None),
        'file_names': clusterer.valid_file_names,
        'cluster_labels': getattr(clusterer, 'cluster_labels', None),
        'n_embeddings': clusterer.n_embeddings,
        'embedding_dim': clusterer.embeddings.shape[1],
        'optimal_k': getattr(clusterer, 'optimal_k', None),
        'scaler': getattr(clusterer, 'scaler', None)
    }
    
    # Save sklearn model separately
    if hasattr(clusterer, 'best_kmeans') and clusterer.best_kmeans is not None:
        joblib.dump(clusterer.best_kmeans, f'{filepath}/kmeans_model.joblib')
        components['has_kmeans'] = True
    else:
        components['has_kmeans'] = False
    
    # Save cluster analysis if available
    if hasattr(clusterer, 'cluster_analysis'):
        # Convert cluster analysis to JSON-serializable format
        cluster_analysis = {}
        for k, v in clusterer.cluster_analysis.items():
            cluster_analysis[str(k)] = {
                'size': int(v['size']),
                'percentage': float(v['percentage']),
                'avg_intra_distance': float(v['avg_intra_distance']),
                'max_intra_distance': float(v['max_intra_distance']),
                'avg_distance_to_center': float(v['avg_distance_to_center']),
                'files': v['files'][:10]  # Save first 10 files only
            }
        
        with open(f'{filepath}/cluster_analysis.json', 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        components['has_cluster_analysis'] = True
    else:
        components['has_cluster_analysis'] = False
    
    # Save main components
    joblib.dump(components, f'{filepath}/components.joblib')
    
    print(f"Clusterer saved to: {filepath}/")
    return filepath

def load_clusterer_components(filepath='clusterer_data'):
    """Load clusterer components and reconstruct object."""
    
    # Load main components
    components = joblib.load(f'{filepath}/components.joblib')
    
    # Create a simple clusterer-like object
    class LoadedClusterer:
        def __init__(self, components, filepath):
            self.embeddings_scaled = components['embeddings']
            self.embeddings = components['embeddings_raw']
            self.valid_file_names = components['file_names']
            self.cluster_labels = components['cluster_labels']
            self.n_embeddings = components['n_embeddings']
            self.embedding_dim = components['embedding_dim']
            self.optimal_k = components['optimal_k']
            self.scaler = components['scaler']
            
            # Load sklearn model if available
            if components['has_kmeans']:
                self.best_kmeans = joblib.load(f'{filepath}/kmeans_model.joblib')
            else:
                self.best_kmeans = None
            
            # Load cluster analysis if available
            if components['has_cluster_analysis']:
                with open(f'{filepath}/cluster_analysis.json', 'r') as f:
                    cluster_analysis_json = json.load(f)
                
                # Convert back to original format
                self.cluster_analysis = {}
                for k, v in cluster_analysis_json.items():
                    self.cluster_analysis[int(k)] = v
            else:
                self.cluster_analysis = None
    
    clusterer = LoadedClusterer(components, filepath)
    print(f"Clusterer loaded from: {filepath}/")
    return clusterer

# if __name__ == "__main__":
    # # Usage:
    # # Save
    # save_clusterer_components(clusterer, '8cluster/NVAE_8_6k')

    # # Load
    # loaded_clusterer = load_clusterer_components('8cluster/NVAE_8_6k')

    # # Verify
    # print(f"Loaded: {loaded_clusterer.n_embeddings} embeddings, {loaded_clusterer.optimal_k} clusters")