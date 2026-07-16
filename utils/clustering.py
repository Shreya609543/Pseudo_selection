import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
import json
import time
from datetime import datetime
from collections import Counter

# Check sklearn version for compatibility
try:
    import sklearn
    sklearn_version = sklearn.__version__
    print(f"Using scikit-learn version: {sklearn_version}")
except:
    print("Could not determine scikit-learn version")

class EmbeddingKMeansClusterer:
    def __init__(self, npz_path):
        """Initialize the clusterer with embeddings from npz file."""
        print("Loading embeddings from npz file...")
        self.data = np.load(npz_path, allow_pickle=True)
        self.file_names = list(self.data.files)
        print(f"Loaded {len(self.file_names)} embeddings")
        
        # Pre-load embeddings into a single array
        print("Pre-loading embeddings into array...")
        embeddings_list = []
        self.valid_indices = []
        
        for i, file_name in enumerate(self.file_names):
            emb = self.data[file_name]
            if emb is not None and len(emb) > 0:
                embeddings_list.append(emb)
                self.valid_indices.append(i)
        
        self.embeddings = np.vstack(embeddings_list)
        self.valid_file_names = [self.file_names[i] for i in self.valid_indices]
        self.n_embeddings = len(self.valid_file_names)
        print(f"Successfully loaded {self.n_embeddings} valid embeddings")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        
        # Normalize embeddings (optional but often helpful for clustering)
        self.scaler = StandardScaler()
        self.embeddings_scaled = self.scaler.fit_transform(self.embeddings)
        
        # Initialize clustering results storage
        self.clustering_results = {}
        self.optimal_k = None
        self.best_kmeans = None
        
    def find_optimal_k(self, k_range=None, max_samples=2000, selection_strategy='max'):
        """Find optimal number of clusters using multiple methods.
        
        Parameters:
        -----------
        k_range : range, optional
            Range of K values to test
        max_samples : int, optional
            Maximum number of samples to use for optimization
        selection_strategy : str, optional
            Strategy for selecting optimal K:
            - 'max': Select maximum K from all methods (default)
            - 'min': Select minimum K from all methods  
            - 'consensus': Select most common K
            - 'silhouette': Use only silhouette score
            - 'elbow': Use only elbow method
            - 'calinski': Use only Calinski-Harabasz score
            - 'davies': Use only Davies-Bouldin score
        """
        if k_range is None:
            k_range = range(2, min(21, self.n_embeddings // 10))
        
        # Use subset for large datasets to speed up computation
        if self.n_embeddings > max_samples:
            indices = np.random.choice(self.n_embeddings, max_samples, replace=False)
            embeddings_subset = self.embeddings_scaled[indices]
            print(f"Using subset of {max_samples} embeddings for optimal K search")
        else:
            embeddings_subset = self.embeddings_scaled
            indices = np.arange(self.n_embeddings)
        
        print(f"Finding optimal K in range {list(k_range)}...")
        
        inertias = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            print(f"Testing K={k}...")
            
            # Fit K-means - NO algorithm parameter
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_subset)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            sil_score = silhouette_score(embeddings_subset, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings_subset, cluster_labels)
            db_score = davies_bouldin_score(embeddings_subset, cluster_labels)
            
            inertias.append(inertia)
            silhouette_scores.append(sil_score)
            calinski_harabasz_scores.append(ch_score)
            davies_bouldin_scores.append(db_score)
            
            print(f"  Inertia: {inertia:.2f}, Silhouette: {sil_score:.3f}, CH: {ch_score:.2f}, DB: {db_score:.3f}")
        
        # Store optimization results
        self.optimization_results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_harabasz_scores,
            'davies_bouldin_scores': davies_bouldin_scores
        }
        
        # Find optimal K using multiple criteria
        optimal_k_elbow = self._find_elbow_point(list(k_range), inertias)
        optimal_k_silhouette = list(k_range)[np.argmax(silhouette_scores)]
        optimal_k_ch = list(k_range)[np.argmax(calinski_harabasz_scores)]
        optimal_k_db = list(k_range)[np.argmin(davies_bouldin_scores)]
        
        # Select optimal K based on strategy
        k_votes = [optimal_k_elbow, optimal_k_silhouette, optimal_k_ch, optimal_k_db]
        method_names = ['Elbow', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
        
        if selection_strategy == 'max':
            self.optimal_k = max(k_votes)
            selected_method = method_names[k_votes.index(self.optimal_k)]
            strategy_description = f"Maximum from {selected_method} method"
        elif selection_strategy == 'min':
            self.optimal_k = min(k_votes)
            selected_method = method_names[k_votes.index(self.optimal_k)]
            strategy_description = f"Minimum from {selected_method} method"
        elif selection_strategy == 'consensus':
            k_counter = Counter(k_votes)
            if len(k_counter.most_common(1)) > 0:
                self.optimal_k = k_counter.most_common(1)[0][0]
                strategy_description = "Consensus (most common)"
            else:
                self.optimal_k = optimal_k_silhouette
                strategy_description = "Silhouette (fallback)"
        elif selection_strategy == 'silhouette':
            self.optimal_k = optimal_k_silhouette
            strategy_description = "Silhouette method only"
        elif selection_strategy == 'elbow':
            self.optimal_k = optimal_k_elbow
            strategy_description = "Elbow method only"
        elif selection_strategy == 'calinski':
            self.optimal_k = optimal_k_ch
            strategy_description = "Calinski-Harabasz method only"
        elif selection_strategy == 'davies':
            self.optimal_k = optimal_k_db
            strategy_description = "Davies-Bouldin method only"
        else:
            # Default to max if unknown strategy
            self.optimal_k = max(k_votes)
            selected_method = method_names[k_votes.index(self.optimal_k)]
            strategy_description = f"Maximum from {selected_method} method (default)"
        
        print(f"\nOptimal K selection:")
        print(f"  Elbow method: K={optimal_k_elbow}")
        print(f"  Silhouette score: K={optimal_k_silhouette}")
        print(f"  Calinski-Harabasz: K={optimal_k_ch}")
        print(f"  Davies-Bouldin: K={optimal_k_db}")
        print(f"  Selected optimal K: {self.optimal_k} ({strategy_description})")
        print(f"  Selection strategy: {selection_strategy}")
        
        # Store the selection strategy for reporting
        self.selection_strategy = selection_strategy
        
        return self.optimal_k
    
    def _find_elbow_point(self, k_values, inertias):
        """Find elbow point using the rate of change method."""
        if len(inertias) < 3:
            return k_values[0]
        
        # Calculate second differences
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        
        # Find the point where the second difference is maximum (steepest change)
        elbow_idx = np.argmax(second_diffs) + 1  # +1 because of double diff
        return k_values[elbow_idx]
    
    def perform_clustering(self, k=None):
        """Perform K-means clustering with specified or optimal K."""
        if k is None:
            if self.optimal_k is None:
                print("Finding optimal K first...")
                self.find_optimal_k()
            k = self.optimal_k
        
        print(f"Performing K-means clustering with K={k}...")
        
        # Perform clustering - NO algorithm parameter
        start_time = time.time()
        print("Creating KMeans instance...")
        self.best_kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10
        )
        
        print("Fitting KMeans to data...")
        self.cluster_labels = self.best_kmeans.fit_predict(self.embeddings_scaled)
        clustering_time = time.time() - start_time
        
        # Calculate clustering metrics
        self.silhouette_avg = silhouette_score(self.embeddings_scaled, self.cluster_labels)
        self.calinski_harabasz = calinski_harabasz_score(self.embeddings_scaled, self.cluster_labels)
        self.davies_bouldin = davies_bouldin_score(self.embeddings_scaled, self.cluster_labels)
        self.inertia = self.best_kmeans.inertia_
        
        # Analyze clusters
        self.cluster_analysis = self._analyze_clusters()
        
        print(f"Clustering completed in {clustering_time:.2f} seconds")
        print(f"Silhouette Score: {self.silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {self.calinski_harabasz:.2f}")
        print(f"Davies-Bouldin Score: {self.davies_bouldin:.4f}")
        print(f"Inertia: {self.inertia:.2f}")
        
        return self.cluster_labels
    
    def _analyze_clusters(self):
        """Analyze cluster characteristics."""
        cluster_analysis = {}
        
        for cluster_id in range(self.best_kmeans.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_embeddings = self.embeddings_scaled[cluster_mask]
            cluster_files = [self.valid_file_names[i] for i in range(len(self.valid_file_names)) if cluster_mask[i]]
            
            # Basic statistics
            cluster_size = np.sum(cluster_mask)
            cluster_center = self.best_kmeans.cluster_centers_[cluster_id]
            
            # Intra-cluster distances
            if cluster_size > 1:
                intra_distances = pairwise_distances(cluster_embeddings)
                avg_intra_distance = np.mean(intra_distances[np.triu_indices_from(intra_distances, k=1)])
                max_intra_distance = np.max(intra_distances)
            else:
                avg_intra_distance = 0
                max_intra_distance = 0
            
            # Distance to center
            distances_to_center = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            avg_distance_to_center = np.mean(distances_to_center)
            
            cluster_analysis[cluster_id] = {
                'size': cluster_size,
                'percentage': (cluster_size / len(self.cluster_labels)) * 100,
                'files': cluster_files[:10],  # Store first 10 files as examples
                'avg_intra_distance': avg_intra_distance,
                'max_intra_distance': max_intra_distance,
                'avg_distance_to_center': avg_distance_to_center,
                'center': cluster_center
            }
        
        return cluster_analysis
    
    def create_optimization_plots(self, save_path='clustering_optimization'):
        """Create plots for K optimization results."""
        if not hasattr(self, 'optimization_results'):
            print("No optimization results available. Run find_optimal_k() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        k_range = self.optimization_results['k_range']
        
        # 1. Elbow plot
        axes[0, 0].plot(k_range, self.optimization_results['inertias'], 'bo-')
        axes[0, 0].set_xlabel('Number of clusters (K)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method for Optimal K')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Highlight optimal K
        if hasattr(self, 'optimal_k'):
            optimal_idx = k_range.index(self.optimal_k)
            axes[0, 0].axvline(x=self.optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Selected K={self.optimal_k}')
            axes[0, 0].legend()
        
        # 2. Silhouette scores
        axes[0, 1].plot(k_range, self.optimization_results['silhouette_scores'], 'go-')
        axes[0, 1].set_xlabel('Number of clusters (K)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Highlight best silhouette score
        best_sil_k = k_range[np.argmax(self.optimization_results['silhouette_scores'])]
        axes[0, 1].axvline(x=best_sil_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_sil_k}')
        axes[0, 1].legend()
        
        # 3. Calinski-Harabasz scores
        axes[1, 0].plot(k_range, self.optimization_results['calinski_harabasz_scores'], 'mo-')
        axes[1, 0].set_xlabel('Number of clusters (K)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Analysis')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Davies-Bouldin scores
        axes[1, 1].plot(k_range, self.optimization_results['davies_bouldin_scores'], 'co-')
        axes[1, 1].set_xlabel('Number of clusters (K)')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].set_title('Davies-Bouldin Analysis (Lower is Better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cluster_visualizations(self, save_path='clustering_results', max_samples_viz=2000):
        """Create comprehensive cluster visualizations."""
        if not hasattr(self, 'cluster_labels'):
            print("No clustering results available. Run perform_clustering() first.")
            return
        
        # Use subset for visualization if dataset is large
        if self.n_embeddings > max_samples_viz:
            viz_indices = np.random.choice(self.n_embeddings, max_samples_viz, replace=False)
            embeddings_viz = self.embeddings_scaled[viz_indices]
            labels_viz = self.cluster_labels[viz_indices]
            print(f"Using subset of {max_samples_viz} embeddings for visualization")
        else:
            embeddings_viz = self.embeddings_scaled
            labels_viz = self.cluster_labels
            viz_indices = np.arange(self.n_embeddings)
        
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. PCA visualization
        print("Creating PCA visualization...")
        ax1 = plt.subplot(2, 3, 1)
        pca = PCA(n_components=2, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings_viz)
        
        scatter = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                            c=labels_viz, cmap='tab10', alpha=0.7, s=20)
        ax1.set_title(f'PCA Visualization (K={self.best_kmeans.n_clusters})')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=ax1)
        
        # 2. t-SNE visualization
        print("Creating t-SNE visualization...")
        ax2 = plt.subplot(2, 3, 2)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_viz)//4))
        embeddings_tsne = tsne.fit_transform(embeddings_viz)
        
        scatter = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                            c=labels_viz, cmap='tab10', alpha=0.7, s=20)
        ax2.set_title(f't-SNE Visualization (K={self.best_kmeans.n_clusters})')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax2)
        
        # 3. Cluster sizes
        ax3 = plt.subplot(2, 3, 3)
        cluster_sizes = [self.cluster_analysis[i]['size'] for i in range(self.best_kmeans.n_clusters)]
        cluster_ids = list(range(self.best_kmeans.n_clusters))
        
        bars = ax3.bar(cluster_ids, cluster_sizes, alpha=0.7, color='skyblue')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Embeddings')
        ax3.set_title('Cluster Sizes')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}', ha='center', va='bottom')
        
        # 4. Intra-cluster distances
        ax4 = plt.subplot(2, 3, 4)
        avg_intra_distances = [self.cluster_analysis[i]['avg_intra_distance'] for i in range(self.best_kmeans.n_clusters)]
        
        ax4.bar(cluster_ids, avg_intra_distances, alpha=0.7, color='lightcoral')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Average Intra-cluster Distance')
        ax4.set_title('Cluster Cohesion')
        ax4.grid(True, alpha=0.3)
        
        # 5. Distances to cluster centers
        ax5 = plt.subplot(2, 3, 5)
        avg_center_distances = [self.cluster_analysis[i]['avg_distance_to_center'] for i in range(self.best_kmeans.n_clusters)]
        
        ax5.bar(cluster_ids, avg_center_distances, alpha=0.7, color='lightgreen')
        ax5.set_xlabel('Cluster ID')
        ax5.set_ylabel('Average Distance to Center')
        ax5.set_title('Distance to Cluster Centers')
        ax5.grid(True, alpha=0.3)
        
        # 6. Silhouette analysis per cluster
        ax6 = plt.subplot(2, 3, 6)
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(embeddings_viz, labels_viz)
        
        y_lower = 10
        for i in range(self.best_kmeans.n_clusters):
            cluster_silhouette_vals = silhouette_vals[labels_viz == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(i / self.best_kmeans.n_clusters)
            ax6.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax6.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax6.set_xlabel('Silhouette coefficient values')
        ax6.set_ylabel('Cluster label')
        ax6.set_title('Silhouette Analysis per Cluster')
        
        # Add average silhouette score line
        ax6.axvline(x=self.silhouette_avg, color="red", linestyle="--", 
                   label=f'Average Score: {self.silhouette_avg:.3f}')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store visualization data
        self.visualization_data = {
            'pca_components': embeddings_pca,
            'tsne_components': embeddings_tsne,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'visualization_indices': viz_indices
        }
    
    def export_clustering_results(self, output_path='clustering_results'):
        """Export clustering results to various formats."""
        if not hasattr(self, 'cluster_labels'):
            print("No clustering results available. Run perform_clustering() first.")
            return
        
        # 1. Export cluster assignments
        cluster_assignments = pd.DataFrame({
            'file_name': self.valid_file_names,
            'cluster_id': self.cluster_labels
        })
        cluster_assignments.to_csv(f'{output_path}_assignments.csv', index=False)
        
        # 2. Export cluster analysis
        cluster_summary = []
        for cluster_id, analysis in self.cluster_analysis.items():
            cluster_summary.append({
                'cluster_id': cluster_id,
                'size': analysis['size'],
                'percentage': analysis['percentage'],
                'avg_intra_distance': analysis['avg_intra_distance'],
                'max_intra_distance': analysis['max_intra_distance'],
                'avg_distance_to_center': analysis['avg_distance_to_center']
            })
        
        cluster_summary_df = pd.DataFrame(cluster_summary)
        cluster_summary_df.to_csv(f'{output_path}_summary.csv', index=False)
        
        # 3. Export detailed results as JSON
        results_dict = {
            'clustering_parameters': {
                'n_clusters': int(self.best_kmeans.n_clusters),
                'algorithm': 'lloyd',  # Default algorithm for compatibility
                'random_state': self.best_kmeans.random_state,
                'optimal_k': int(self.optimal_k) if self.optimal_k else None,
                'selection_strategy': getattr(self, 'selection_strategy', 'max')
            },
            'metrics': {
                'silhouette_score': float(self.silhouette_avg),
                'calinski_harabasz_score': float(self.calinski_harabasz),
                'davies_bouldin_score': float(self.davies_bouldin),
                'inertia': float(self.inertia)
            },
            'cluster_centers': self.best_kmeans.cluster_centers_.tolist(),
            'cluster_analysis': {str(k): {
                'size': int(v['size']),
                'percentage': float(v['percentage']),
                'avg_intra_distance': float(v['avg_intra_distance']),
                'max_intra_distance': float(v['max_intra_distance']),
                'avg_distance_to_center': float(v['avg_distance_to_center']),
                'example_files': v['files']
            } for k, v in self.cluster_analysis.items()}
        }
        
        # Include optimization results if available
        if hasattr(self, 'optimization_results'):
            results_dict['optimization_results'] = {
                'k_range': self.optimization_results['k_range'],
                'inertias': [float(x) for x in self.optimization_results['inertias']],
                'silhouette_scores': [float(x) for x in self.optimization_results['silhouette_scores']],
                'calinski_harabasz_scores': [float(x) for x in self.optimization_results['calinski_harabasz_scores']],
                'davies_bouldin_scores': [float(x) for x in self.optimization_results['davies_bouldin_scores']]
            }
        
        with open(f'{output_path}_detailed.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        # 4. Export cluster centers
        centers_df = pd.DataFrame(self.best_kmeans.cluster_centers_, 
                                columns=[f'dim_{i}' for i in range(self.embeddings.shape[1])])
        centers_df.index.name = 'cluster_id'
        centers_df.to_csv(f'{output_path}_centers.csv')
        
        print(f"Clustering results exported to:")
        print(f"  - {output_path}_assignments.csv (cluster assignments)")
        print(f"  - {output_path}_summary.csv (cluster summary)")
        print(f"  - {output_path}_detailed.json (detailed results)")
        print(f"  - {output_path}_centers.csv (cluster centers)")
    
    def generate_clustering_report(self, output_file='clustering_report.txt'):
        """Generate a comprehensive clustering analysis report."""
        if not hasattr(self, 'cluster_labels'):
            print("No clustering results available. Run perform_clustering() first.")
            return
        
        report_content = f"""
K-MEANS CLUSTERING ANALYSIS REPORT
{'='*80}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of embeddings: {self.n_embeddings}
Embedding dimension: {self.embeddings.shape[1]}
Number of clusters: {self.best_kmeans.n_clusters}

CLUSTERING PARAMETERS
{'='*80}
Algorithm: lloyd (default)
Random state: {self.best_kmeans.random_state}
Number of initializations: {self.best_kmeans.n_init}
Optimal K (selected): {self.optimal_k}
Selection strategy: {getattr(self, 'selection_strategy', 'max')}

CLUSTERING QUALITY METRICS
{'='*80}
Silhouette Score: {self.silhouette_avg:.4f}
  Interpretation: Higher is better (range: -1 to 1)
  Your score: {'Excellent (>0.7)' if self.silhouette_avg > 0.7 else 'Good (0.5-0.7)' if self.silhouette_avg > 0.5 else 'Fair (0.25-0.5)' if self.silhouette_avg > 0.25 else 'Poor (<0.25)'}

Calinski-Harabasz Score: {self.calinski_harabasz:.2f}
  Interpretation: Higher is better (no upper bound)
  
Davies-Bouldin Score: {self.davies_bouldin:.4f}
  Interpretation: Lower is better (0 is best)
  
Inertia (Within-cluster sum of squares): {self.inertia:.2f}
  Interpretation: Lower is better for fixed K

"""
        
        # Add optimization results if available
        if hasattr(self, 'optimization_results'):
            report_content += f"""
K OPTIMIZATION ANALYSIS
{'='*80}
Tested K range: {self.optimization_results['k_range']}

Method recommendations:
"""
            k_range = self.optimization_results['k_range']
            optimal_k_silhouette = k_range[np.argmax(self.optimization_results['silhouette_scores'])]
            optimal_k_ch = k_range[np.argmax(self.optimization_results['calinski_harabasz_scores'])]
            optimal_k_db = k_range[np.argmin(self.optimization_results['davies_bouldin_scores'])]
            
            report_content += f"  Silhouette method: K = {optimal_k_silhouette}\n"
            report_content += f"  Calinski-Harabasz method: K = {optimal_k_ch}\n"
            report_content += f"  Davies-Bouldin method: K = {optimal_k_db}\n"
            report_content += f"  Selected: K = {self.optimal_k}\n\n"
        
        # Add cluster analysis
        report_content += f"""
CLUSTER ANALYSIS
{'='*80}

{'Cluster':<8} {'Size':<8} {'%':<8} {'Avg Intra Dist':<15} {'Avg to Center':<15} {'Examples':<30}
{'-'*8} {'-'*8} {'-'*8} {'-'*15} {'-'*15} {'-'*30}
"""
        
        for cluster_id in range(self.best_kmeans.n_clusters):
            analysis = self.cluster_analysis[cluster_id]
            examples = ', '.join(analysis['files'][:3])  # First 3 examples
            if len(analysis['files']) > 3:
                examples += '...'
            
            report_content += f"{cluster_id:<8} {analysis['size']:<8} {analysis['percentage']:<8.1f} {analysis['avg_intra_distance']:<15.4f} {analysis['avg_distance_to_center']:<15.4f} {examples:<30}\n"
        
        # Add insights and recommendations
        insights = []
        
        # Cluster size analysis
        cluster_sizes = [self.cluster_analysis[i]['size'] for i in range(self.best_kmeans.n_clusters)]
        size_std = np.std(cluster_sizes)
        size_mean = np.mean(cluster_sizes)
        
        if size_std / size_mean > 0.5:
            insights.append("📊 UNBALANCED CLUSTERS: Large variation in cluster sizes suggests potential imbalanced clustering or natural data structure.")
        
        # Silhouette score analysis
        if self.silhouette_avg > 0.7:
            insights.append("✅ EXCELLENT CLUSTERING: High silhouette score indicates well-separated and cohesive clusters.")
        elif self.silhouette_avg < 0.25:
            insights.append("⚠️  POOR CLUSTERING: Low silhouette score suggests overlapping or poorly defined clusters. Consider different K or preprocessing.")
        
        # Cohesion analysis
        avg_intra_distances = [self.cluster_analysis[i]['avg_intra_distance'] for i in range(self.best_kmeans.n_clusters)]
        if np.std(avg_intra_distances) > np.mean(avg_intra_distances):
            insights.append("📈 VARIABLE COHESION: Some clusters are much more cohesive than others. Consider sub-clustering large clusters.")
        
        if insights:
            report_content += f"""

KEY INSIGHTS
{'='*80}

"""
            for insight in insights:
                report_content += f"{insight}\n\n"
        
        # Add recommendations
        recommendations = []
        
        if self.silhouette_avg < 0.5:
            recommendations.append("Consider trying different values of K or using different clustering algorithms (DBSCAN, hierarchical clustering).")
        
        if size_std / size_mean > 0.7:
            recommendations.append("Large clusters might benefit from sub-clustering to reveal finer structure.")
        
        recommendations.append("Use the cluster assignments to analyze patterns in your original data/files.")
        recommendations.append("Consider visualizing clusters using PCA or t-SNE for better understanding.")
        recommendations.append("Validate clusters by examining the actual files/data points in each cluster.")
        
        if recommendations:
            report_content += f"""
RECOMMENDATIONS
{'='*80}

"""
            for i, rec in enumerate(recommendations, 1):
                report_content += f"{i}. {rec}\n\n"
        
        # Add technical details
        report_content += f"""
TECHNICAL DETAILS
{'='*80}

Data preprocessing:
- Embeddings were standardized (zero mean, unit variance)
- No dimensionality reduction applied before clustering

Cluster centers shape: {self.best_kmeans.cluster_centers_.shape}
Convergence: {'Yes' if hasattr(self.best_kmeans, 'n_iter_') else 'Unknown'}

Files generated:
- clustering_results_assignments.csv: Individual cluster assignments
- clustering_results_summary.csv: Cluster summary statistics  
- clustering_results_detailed.json: Complete results in JSON format
- clustering_results_centers.csv: Cluster center coordinates
- clustering_optimization_metrics.png: K optimization plots
- clustering_results_visualizations.png: Cluster visualizations
- {output_file}: This analysis report

"""
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Clustering analysis report generated: {output_file}")
        return output_file
    
    def get_cluster_members(self, cluster_id):
        """Get all members of a specific cluster."""
        if not hasattr(self, 'cluster_labels'):
            print("No clustering results available. Run perform_clustering() first.")
            return []
        
        cluster_mask = self.cluster_labels == cluster_id
        cluster_files = [self.valid_file_names[i] for i in range(len(self.valid_file_names)) if cluster_mask[i]]
        return cluster_files
    
    def find_nearest_neighbors_in_cluster(self, file_name, n_neighbors=5):
        """Find nearest neighbors of a file within its cluster."""
        if not hasattr(self, 'cluster_labels'):
            print("No clustering results available. Run perform_clustering() first.")
            return []
        
        try:
            file_idx = self.valid_file_names.index(file_name)
        except ValueError:
            print(f"File {file_name} not found in embeddings.")
            return []
        
        cluster_id = self.cluster_labels[file_idx]
        cluster_mask = self.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) <= 1:
            return []
        
        # Get embeddings for this cluster
        file_embedding = self.embeddings_scaled[file_idx].reshape(1, -1)
        cluster_embeddings = self.embeddings_scaled[cluster_indices]
        
        # Calculate distances
        distances = pairwise_distances(file_embedding, cluster_embeddings)[0]
        
        # Get nearest neighbors (excluding the file itself)
        neighbor_indices = np.argsort(distances)[1:n_neighbors+1]
        neighbors = [(self.valid_file_names[cluster_indices[idx]], distances[idx]) 
                    for idx in neighbor_indices]
        
        return neighbors

def run_complete_clustering_analysis(npz_path, max_k=20, max_samples=2000, selection_strategy='max',out_path = ""):
    """Run complete K-means clustering analysis workflow.
    
    Parameters:
    -----------
    npz_path : str
        Path to NPZ file containing embeddings
    max_k : int
        Maximum K to test (default: 20)
    max_samples : int  
        Maximum samples for K optimization (default: 2000)
    selection_strategy : str
        K selection strategy: 'max', 'min', 'consensus', 'silhouette', 'elbow', 'calinski', 'davies'
    """
    print("Starting Complete K-means Clustering Analysis...")
    print("="*60)
    
    # Initialize clusterer
    clusterer = EmbeddingKMeansClusterer(npz_path)
    
    # Find optimal K
    start_time = time.time()
    optimal_k = clusterer.find_optimal_k(k_range=range(2, max_k+1), max_samples=max_samples, selection_strategy=selection_strategy)
    optimization_time = time.time() - start_time
    
    # Perform clustering with optimal K
    start_time = time.time()
    cluster_labels = clusterer.perform_clustering(k=optimal_k)
    clustering_time = time.time() - start_time
    
    # Create visualizations
    clusterer.create_optimization_plots()
    clusterer.create_cluster_visualizations()
    
    # Export results
    clusterer.export_clustering_results(output_path = out_path)
    
    # Generate report
    report_file = clusterer.generate_clustering_report()
    
    # Print summary
    print(f"\n{'='*60}")
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*60)
    print(f"Optimal K: {optimal_k} (using '{selection_strategy}' strategy)")
    print(f"Silhouette Score: {clusterer.silhouette_avg:.4f}")
    print(f"Number of clusters: {clusterer.best_kmeans.n_clusters}")
    
    print(f"\nCluster sizes:")
    for i in range(clusterer.best_kmeans.n_clusters):
        size = clusterer.cluster_analysis[i]['size']
        percentage = clusterer.cluster_analysis[i]['percentage']
        print(f"  Cluster {i}: {size} embeddings ({percentage:.1f}%)")
    
    print(f"\nTiming:")
    print(f"  K optimization: {optimization_time:.2f} seconds")
    print(f"  Final clustering: {clustering_time:.2f} seconds")
    
    print(f"\n🎉 Clustering Analysis Complete!")
    print(f"📊 Report: {report_file}")
    
    return clusterer

# Example usage functions
def analyze_specific_cluster(clusterer, cluster_id):
    """Analyze a specific cluster in detail."""
    if not hasattr(clusterer, 'cluster_labels'):
        print("No clustering results available.")
        return
    
    cluster_members = clusterer.get_cluster_members(cluster_id)
    analysis = clusterer.cluster_analysis[cluster_id]
    
    print(f"\nDETAILED ANALYSIS - CLUSTER {cluster_id}")
    print("="*50)
    print(f"Size: {analysis['size']} embeddings ({analysis['percentage']:.1f}%)")
    print(f"Average intra-cluster distance: {analysis['avg_intra_distance']:.4f}")
    print(f"Average distance to center: {analysis['avg_distance_to_center']:.4f}")
    print(f"\nMembers (showing first 20):")
    for i, member in enumerate(cluster_members[:20]):
        print(f"  {i+1:2d}. {member}")
    
    if len(cluster_members) > 20:
        print(f"  ... and {len(cluster_members) - 20} more")

if __name__ == "__main__":
    # Example usage
    npz_file_path = "path/to/pseudo_profiles.npz"  # Replace with your file path
    out_dir = ""
    # Use maximum K selection strategy (your preference)
    clusterer = run_complete_clustering_analysis(
        npz_file_path, 
        max_k=8, 
        max_samples=6000,
        out_path= out_dir,
        selection_strategy='max'  # Options: 'max', 'min', 'consensus', 'silhouette', 'elbow', 'calinski', 'davies'
    )
    
    # Example: Analyze cluster 0
    # analyze_specific_cluster(clusterer, 0)
    
    # Example: Find neighbors of a specific file
    # neighbors = clusterer.find_nearest_neighbors_in_cluster("your_file_name.ext", n_neighbors=5)
    # print(f"Nearest neighbors: {neighbors}")