import numpy as np
import joblib
from typing import List, Tuple, Dict, Any
import warnings
from pathlib import Path
from utils.save_load import load_clusterer_components


class EmbeddingSelector:
    """
    Production-ready embedding selector for finding dissimilar embeddings
    based on cluster analysis and Davies-Bouldin similarity scoring.
    Compatible with both regular joblib clusterers and LoadedClusterer objects.
    """
    
    def __init__(self, clusterer_path: str = None, secondary_clusterer_path : str = "", used_indexes: List[int] = [],  clusterer_obj: Any = None,  use_custom_loader: bool = True):
        """
        Initialize with either clusterer path or object.
        
        Args:
            clusterer_path: Path to clusterer (either .joblib file or directory)
            used_indexes: List of already used filename values (will be modified)
            clusterer_obj: Pre-loaded clusterer object
            use_custom_loader: Whether to try custom loader first for directories
        """
        if clusterer_path:
            clusterer_path = Path(clusterer_path)
            
            # Try custom loader first if it's a directory and custom loader is available
            if (clusterer_path.is_dir() and use_custom_loader and 
                load_clusterer_components is not None):
                try:
                    self.clusterer = load_clusterer_components(str(clusterer_path))
                    print(f"Loaded using custom loader from: {clusterer_path}")
                except Exception as e:
                    print(f"Custom loader failed: {e}")
                    raise
            
            # Fallback to regular joblib loading
            elif clusterer_path.is_file() and clusterer_path.suffix == '.joblib':
                self.clusterer = joblib.load(clusterer_path)
                print(f"Loaded using joblib from: {clusterer_path}")
            
            else:
                raise ValueError(f"Invalid clusterer path: {clusterer_path}")
                
        elif clusterer_obj:
            self.clusterer = clusterer_obj
        else:
            raise ValueError("Either clusterer_path or clusterer_obj must be provided")
            
        self.embeddings = self.clusterer.embeddings_scaled
        self.file_names = self.clusterer.valid_file_names
        self.cluster_labels = self.clusterer.cluster_labels
        self.cluster_centers = self.clusterer.best_kmeans.cluster_centers_
        self.n_clusters = self.clusterer.best_kmeans.n_clusters
        
        # Pre-calculate cluster statistics for efficiency
        self._calculate_cluster_stats()
        
        # Convert used_indexes from filename values to array indices for internal logic
        self.used_array_indices = []
        for filename_val in used_indexes:
            try:
                # Find the array index where filename equals the used filename
                array_idx = next(i for i, fname in enumerate(self.file_names) 
                            if (fname.isdigit() and int(fname) == filename_val) or fname == str(filename_val))
                self.used_array_indices.append(array_idx)
            except StopIteration:
                # If filename not found, skip it
                continue
        
    
    def _calculate_cluster_stats(self):
        """Pre-calculate cluster statistics."""
        self.cluster_stats = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_points = self.embeddings[cluster_mask]
            cluster_center = self.cluster_centers[cluster_id]
            
            if len(cluster_points) > 0:
                distances_to_center = np.linalg.norm(cluster_points - cluster_center, axis=1)
                within_scatter = np.mean(distances_to_center)
            else:
                within_scatter = 0.0
            
            self.cluster_stats[cluster_id] = {
                'center': cluster_center,
                'within_scatter': within_scatter,
                'size': len(cluster_points),
                'indices': np.where(cluster_mask)[0].tolist()
            }
    
    def _find_similar_clusters(self, new_embedding: np.ndarray, top_k: int = 3) -> List[int]:
        """
        Find top-k most similar clusters using Davies-Bouldin method.
        
        Args:
            new_embedding: New embedding vector
            top_k: Number of similar clusters to return
            
        Returns:
            List of cluster IDs sorted by similarity (most similar first)
        """
        if new_embedding.ndim == 2:
            new_embedding = new_embedding.flatten()
        
        cluster_scores = {}
        for cluster_id in range(self.n_clusters):
            cluster_center = self.cluster_stats[cluster_id]['center']
            within_scatter = self.cluster_stats[cluster_id]['within_scatter']
            
            distance_to_center = np.linalg.norm(new_embedding - cluster_center)
            
            # Davies-Bouldin inspired normalized score
            if within_scatter == 0:
                score = distance_to_center
            else:
                score = distance_to_center / (within_scatter + 1e-8)
            
            cluster_scores[cluster_id] = score
        
        # Return top-k clusters with lowest scores (most similar)
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1])
        return [cluster_id for cluster_id, _ in sorted_clusters[:top_k]]
    
    def _find_dissimilar_embeddings(self, new_embedding: np.ndarray, 
                                   exclude_clusters: List[int], 
                                   n_results: int = 50) -> List[Tuple[int, float, int]]:
        """
        Find most dissimilar embeddings excluding specified clusters.
        
        Args:
            new_embedding: New embedding vector
            exclude_clusters: Clusters to exclude from search
            n_results: Number of dissimilar embeddings to return
            
        Returns:
            List of tuples (embedding_index, distance, cluster_id)
        """
        if new_embedding.ndim == 2:
            new_embedding = new_embedding.flatten()
        
        # Create candidate mask (exclude specified clusters)
        candidate_mask = np.ones(len(self.embeddings), dtype=bool)
        for cluster_id in exclude_clusters:
            candidate_mask &= (self.cluster_labels != cluster_id)
        
        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            return []
        
        candidate_embeddings = self.embeddings[candidate_indices]
        
        # Calculate cosine distances
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(new_embedding.reshape(1, -1), candidate_embeddings)[0]
        
        # Get most dissimilar (largest distances)
        dissimilar_indices = np.argsort(distances)[::-1][:n_results]
        
        results = []
        for idx in dissimilar_indices:
            actual_idx = candidate_indices[idx]
            cluster_id = self.cluster_labels[actual_idx]
            results.append((actual_idx, distances[idx], cluster_id))
        
        return results
    
    def select_embeddings(self, new_embedding: np.ndarray,  
                         n_select: int = 8) -> List[int]:
        """
        Main function to select dissimilar embeddings based on specified criteria.
        
        Args:
            new_embedding: New embedding vector to compare against
            n_select: Number of embeddings to select (default: 8)
            
        Returns:
            List of selected filename values (not array indices)
        """

        
        # Step 1: Find top 3 similar clusters
        similar_clusters = self._find_similar_clusters(new_embedding, top_k=3)
        
        # Step 2: Find dissimilar embeddings excluding similar clusters
        dissimilar_embeddings = self._find_dissimilar_embeddings(
            new_embedding, exclude_clusters=similar_clusters, n_results=100
        )
        
        # Step 3: Apply selection criteria
        selected_array_indices = []
        used_clusters = set()
        
        # Filter out used indices and prioritize different clusters
        available_embeddings = [
            (idx, dist, cluster_id) for idx, dist, cluster_id in dissimilar_embeddings
            if idx not in self.used_array_indices
        ]
        
        # Phase 1: One embedding per cluster
        for idx, dist, cluster_id in available_embeddings:
            if len(selected_array_indices) >= n_select:
                break
            if cluster_id not in used_clusters:
                selected_array_indices.append(idx)
                used_clusters.add(cluster_id)
        
        # Phase 2: If still need more, allow multiple from same cluster
        if len(selected_array_indices) < n_select:
            for idx, dist, cluster_id in available_embeddings:
                if len(selected_array_indices) >= n_select:
                    break
                if idx not in selected_array_indices:
                    selected_array_indices.append(idx)
        
        # Phase 3: If still need more, include from all clusters (excluding used_indexes)
        if len(selected_array_indices) < n_select:
            all_dissimilar = self._find_dissimilar_embeddings(
                new_embedding, exclude_clusters=[], n_results=200
            )
            
            for idx, dist, cluster_id in all_dissimilar:
                if len(selected_array_indices) >= n_select:
                    break
                if idx not in self.used_array_indices and idx not in selected_array_indices:
                    selected_array_indices.append(idx)
        
        # Convert selected array indices back to filename values
        selected_filename_values = []
        for array_idx in selected_array_indices[:n_select]:
            filename_val = int(self.file_names[array_idx]) if self.file_names[array_idx].isdigit() else array_idx
            selected_filename_values.append(filename_val)
        
        # Update used_indexes with selected filename values
        used_indexes.extend(selected_filename_values)
        
        return selected_filename_values
    
    def get_embedding_info(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get detailed information about selected embeddings.
        
        Args:
            indices: List of embedding indices
            
        Returns:
            List of dictionaries containing embedding information
        """
        info = []
        for idx in indices:
            # Convert filename to int to match the index (assuming filenames are 1-indexed)
            filename_as_int = int(self.file_names[idx]) if self.file_names[idx].isdigit() else idx
            
            info.append({
                'index': filename_as_int,  # Use filename as index to match
                'file_name': self.file_names[idx],
                'cluster_id': self.cluster_labels[idx],
                'cluster_size': self.cluster_stats[self.cluster_labels[idx]]['size']
            })
        return info


# Convenience function for single-use scenarios
def select_dissimilar_embeddings(clusterer_path: str, 
                                new_embedding: np.ndarray,
                                used_indexes: List[int],
                                n_select: int = 8,
                                use_custom_loader: bool = True) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Convenience function to select dissimilar embeddings in one call.
    
    Args:
        clusterer_path: Path to clusterer (directory or .joblib file)
        new_embedding: New embedding vector
        used_indexes: List of already used indices (will be modified)
        n_select: Number of embeddings to select
        use_custom_loader: Whether to try custom loader for directories
        
    Returns:
        Tuple of (selected_indices, embedding_info)
    """
    selector = EmbeddingSelector(
        clusterer_path=clusterer_path, 
        use_custom_loader=use_custom_loader
    )
    selected_indices = selector.select_embeddings(new_embedding, used_indexes, n_select)
    embedding_info = selector.get_embedding_info(selected_indices)
    
    return selected_indices, embedding_info
    
    def select_embeddings_nvae_gan(self, new_embedding: np.ndarray, ecapa: bool = True, n_select: int = 8) -> List[int]:
    if ecapa:
        new_embedding = new_embedding[-192:]  # Ensure using ECAPA dimensions
    try: 
        selected_indices = self.select_embeddings(
            new_embedding=new_embedding,
            n_select=n_select  # Number of embeddings to select
        )
    except Exception as e:
        # If it fails, use the secondary clusterer
        try :
            if self.secondary_clusterer_path:
                secondary_selector = EmbeddingSelector(
                    clusterer_path=self.secondary_clusterer_path,
                    use_custom_loader=True
                )
                selected_indices = secondary_selector.select_embeddings(
                    new_embedding=new_embedding,
                    n_select=n_select  # Number of embeddings to select
                )
            else:
                self.used_array_indices = []
                selected_indices = self.select_embeddings(
                    new_embedding=new_embedding,
                    n_select=n_select  # Number of embeddings to select
                )
        except Exception as e:      
            raise e
    
    return selected_indices


if __name__ == "__main__":
    
    cluster_path = ""
    # Selected embeddings
    used_indexes = [] 
    # Method 1: Using your custom loader (for directories)
    selector = EmbeddingSelector(clusterer_path=cluster_path)

    # Method 2: With pre-loaded clusterer object
    from utils.save_load import load_clusterer_components
    clusterer = load_clusterer_components(cluster_path)
    selector = EmbeddingSelector(clusterer_obj=clusterer,         used_indexes=used_indexes)  # This gets modified)

    # Your existing used indexes
    new_embed = np.random.randn(704)  # Your new embedding

    selected_indices = selector.select_embeddings(
        new_embedding=new_embed,
        n_select=8
    )

    # Get detailed info about selections
    embedding_info = selector.get_embedding_info(selected_indices)

    # Or use convenience function with custom loader
    selected_indices, embedding_info = select_dissimilar_embeddings(
        clusterer_path=cluster_path,  # Your directory path
        new_embedding=new_embed,
        used_indexes=used_indexes,
        n_select=8
    )

    # Or with regular joblib file
    selected_indices, embedding_info = select_dissimilar_embeddings(
        clusterer_path='path/to/clusterer.joblib',
        new_embedding=new_embed,
        used_indexes=used_indexes,
        n_select=8,
        use_custom_loader=False
    )