import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 0. Set Random Seed for Reproducibility
seed_value = 42
np.random.seed(seed_value) 
sc.settings.random_state = seed_value 
print(f"Random seed set to: {seed_value}, ensuring experiment reproducibility.")

# Set Scanpy default parameters and output directory
sc.settings.verbosity = 0 
sc.set_figure_params(dpi=200, figsize=(6, 6), facecolor='white')
output_dir = 'UMAP_HPO_Results'
os.makedirs(output_dir, exist_ok=True)
print(f"Analysis results will be saved in directory: {output_dir}")

# File paths
data_path = r"C:\Users\Jilma\Desktop\TCGA.BRCA.sampleMap_AgilentG4502A_07_3" 
base_filename = os.path.basename(data_path) 

# 1. Data Loading and Initial Cleaning
try:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: File path does not exist or file is missing: {data_path}")

    print(f"Starting data loading: {data_path}")
    
    # Try reading with '\t' as delimiter
    try:
        df_raw = pd.read_csv(data_path, index_col=0, header=0, sep='\t')
    except Exception:
        df_raw = pd.read_csv(data_path, index_col=0, header=0) # Default comma
    
    # Cleaning Step: Drop genes (rows) containing any NaN/missing values
    if df_raw.isnull().any().any():
        initial_gene_count = df_raw.shape[0]
        # Drop rows (genes) with any NaN
        df_raw = df_raw.dropna(axis=0, how='any') 
        removed_gene_count = initial_gene_count - df_raw.shape[0]
        if removed_gene_count > 0:
            print(f"Warning: NaN missing values found. Removed {removed_gene_count} genes (rows) containing missing values.")
    
    # Cleaning Step: Remove non-expression metadata columns
    N_META_COLS_TO_DROP = 4
    if df_raw.shape[1] > N_META_COLS_TO_DROP:
        # Get the names of the first N columns to remove
        meta_cols_to_drop = df_raw.columns[:N_META_COLS_TO_DROP]
        df_raw = df_raw.drop(columns=meta_cols_to_drop)
        print(f"Removed first {N_META_COLS_TO_DROP} metadata columns.")
    else:
        print("Not enough metadata columns detected for removal; skipping metadata drop step.")
    
    # Crucial step: Transpose data to (Samples x Genes) format
    df = df_raw.T 
    df.index.name = 'Sample_ID'
    
    if df.shape[0] == 0:
        raise ValueError("Data loading failed: Sample count is 0. Check delimiter, header, and index column.")
        
    # Convert data to float type
    df = df.astype(float)
    
    print(f"Data shape (Samples x Genes): {df.shape}")
    
except Exception as e:
    print("\nData Loading Failed.")
    print(f"Error details: {e}")
    exit()


# Gene Filtering: Select Top 2000 Highest Variance Genes
N_TOP_GENES = 2000
current_gene_count = df.shape[1]
print(f"\nInitial gene count = {current_gene_count}")

if current_gene_count > N_TOP_GENES:
    # 1. Calculate variance for each gene (column)
    gene_variances = df.var(axis=0)
    
    # 2. Find the names of the top N genes with the highest variance
    top_genes_names = gene_variances.nlargest(N_TOP_GENES).index
    
    # 3. Filter DataFrame to keep only high-variance genes
    df = df[top_genes_names]
    print(f"Filtered gene count = {df.shape[1]} (Top {N_TOP_GENES} highest variance)")

    # 4. Export filtered data
    output_filtered_path = os.path.join(output_dir, f"top{N_TOP_GENES}_high_var_genes.csv")
    df.to_csv(output_filtered_path, index=True) 
    print(f"Filtered gene expression data saved to {output_filtered_path}")
else:
    print("Gene count is less than 2000, skipping high variance filtering.")

# Convert DataFrame to Scanpy AnnData object
adata = sc.AnnData(df)

# 2. Preprocessing and Fixed Dimensionality Reduction Steps
print("Preprocessing (Standardization & PCA)")

# Standardization (Z-score)
sc.pp.scale(adata, max_value=10) 

# PCA Dimensionality Reduction (Fixed)
n_pca = min(50, df.shape[1] - 1) 
sc.tl.pca(adata, n_comps=n_pca) 
n_pcs_use = min(30, n_pca)


# 3. Hyperparameter Optimization Search Space
N_NEIGHBORS_RANGE = [10, 20, 30]  
RESOLUTION_RANGE = [0.2, 0.5, 0.8, 1.2] 

best_score = -1
best_params = {}
hpo_results = []
total_runs = len(N_NEIGHBORS_RANGE) * len(RESOLUTION_RANGE)
run_count = 0

print(f"Starting Hyperparameter Optimization (Total {total_runs} runs)")

# 4. Optimization Loop (Manual Grid Search)

for n_neigh in N_NEIGHBORS_RANGE:
    for resolution in RESOLUTION_RANGE:
        run_count += 1
        print(f"[{run_count}/{total_runs}] Testing N_Neighbors={n_neigh}, Resolution={resolution}...")
        
        # Define key for the current neighbors graph
        current_neighbors_key = f'neighbors_{n_neigh}' 
        
        # 4.1 Core: Compute Neighbors Graph
        sc.pp.neighbors(
            adata, 
            n_neighbors=n_neigh, 
            n_pcs=n_pcs_use, 
            key_added=current_neighbors_key 
        ) 
        
        # 4.2 UMAP Dimensionality Reduction
        sc.tl.umap(
            adata, 
            n_components=2, 
            neighbors_key=current_neighbors_key
        ) 
        
        # 4.3 Leiden Clustering
        sc.tl.leiden(
            adata, 
            resolution=resolution, 
            neighbors_key=current_neighbors_key, 
            key_added='temp_clusters' 
        )
        
        # 4.4 Evaluate Clustering Quality (using Silhouette Score)
        try:
            # Ensure more than 1 cluster is found
            if len(adata.obs['temp_clusters'].unique()) > 1:
                # Use adata.X for silhouette score calculation
                score = silhouette_score(
                    adata.X, 
                    adata.obs['temp_clusters'].astype('category')
                )
            else:
                score = -999 
        except Exception:
            score = -999 

        # 4.5 Record and Update Best Results
        hpo_results.append({
            'n_neighbors': n_neigh,
            'resolution': resolution,
            'silhouette_score': score
        })

        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neigh, 'resolution': resolution, 'score': score}
            # Store best cluster labels
            adata.obs['best_clusters'] = adata.obs['temp_clusters'].astype('category').copy() 
            adata.obsm['X_best_umap'] = adata.obsm['X_umap'].copy()


# 5. Final Results and Visualization

print("\n" + "="*80)
print(f"Hyperparameter Optimization Complete! Best Parameter Set:")
print(f"   N_Neighbors: {best_params.get('n_neighbors')}")
print(f"   Resolution: {best_params.get('resolution')}")
print(f"   Silhouette Score: {best_params.get('score'):.4f}")
print("="*80)

if 'best_clusters' in adata.obs:
    # 5.0 Cluster Evaluation Metrics: DBI, CH, Silhouette (Optimal K)
    # Use data from the PCA space for evaluation
    X_features = adata.obsm['X_pca'][:, :n_pcs_use] 
    labels = adata.obs['best_clusters']
    
    # Ensure number of clusters > 1 and data points >= cluster count
    if len(labels.unique()) > 1 and len(X_features) >= len(labels.unique()):
        
        # Calinski-Harabasz Index (CH) - Higher is better
        ch_final = calinski_harabasz_score(X_features, labels)
        
        # Davies-Bouldin Index (DBI) - Lower is better
        dbi_final = davies_bouldin_score(X_features, labels)
        
        # Mean Silhouette Score (Sil) - Closer to 1 is better
        mean_sil_final = best_score
        
        print("\n" + "#"*80)
        print("Cluster Evaluation Metrics (Assessed in PCA space for optimal K)")
        print(f"Davies–Bouldin Index (DBI) for k*: {dbi_final:.3f} (Lower is better)")
        print(f"Calinski–Harabasz Index (CH) for k*: {ch_final:.3f} (Higher is better)")
        print(f"Mean Silhouette Score (Sil) for k*: {mean_sil_final:.3f} (Closer to 1 is better)")
        print("#"*80)
    else:
        print("Cannot calculate DBI/CH metrics: Insufficient number of clusters or samples.")
        
    # 5.1 Save HPO Results Table
    hpo_df = pd.DataFrame(hpo_results).sort_values(by='silhouette_score', ascending=False)
    hpo_df.to_csv(os.path.join(output_dir, f'{base_filename}_HPO_Scores.csv'), index=False)
    print(f"HPO results table saved to {output_dir}")

    # 5.2 Plot Best Result UMAP
    sc.pl.umap(
        adata, 
        color='best_clusters', 
        title=f'UMAP Clustering (Best: N={best_params["n_neighbors"]}, R={best_params["resolution"]})',
        frameon=False, 
        save=f'_{base_filename}_best_clustering.png',
        show=False
    )
    plt.savefig(os.path.join(output_dir, f'{base_filename}_best_clustering.png'))
    print("Generated and saved UMAP clustering plot for best parameters.")

    # 6. Export Cluster Data
    print("\nStarting Export of Cluster-Grouped Data")
    
    # 6.1 Define new export directory
    cluster_export_dir = os.path.join(output_dir, 'Individual_Clusters_Data')
    os.makedirs(cluster_export_dir, exist_ok=True)
    
    # 6.2 Extract scaled gene expression data
    df_expression = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    
    # 6.3 Extract best cluster labels
    clusters = adata.obs['best_clusters'].copy()
    
    # 6.4 Add cluster labels to the expression data
    df_clustered = df_expression.copy()
    df_clustered['Cluster'] = clusters
    
    # 6.5 Determine the number of clusters
    best_k = len(clusters.unique())
    print(f"Detected {best_k} optimal clusters.")
    
    # 6.6 Split and export data
    for cluster_id, df_subset in df_clustered.groupby('Cluster'):
        
        # 1. Create filename
        try:
            # Try converting to integer for 1-based naming
            cluster_num = int(cluster_id) + 1 
        except ValueError:
            # Fallback for non-numeric cluster IDs
            cluster_num = str(cluster_id)

        # Ensure .csv extension is used
        cluster_name = f"{base_filename}_cluster{cluster_num}.csv"
        
        # 2. Remove the 'Cluster' column
        df_to_save = df_subset.drop(columns=['Cluster'])
        
        # 3. Export to CSV. index=False means Sample IDs (index) are not included as a column.
        df_to_save.to_csv(os.path.join(cluster_export_dir, cluster_name), index=False) 
        
        print(f"Saved cluster data: {cluster_name} (Sample count: {len(df_to_save)})")

    print("\nCluster-grouped data export complete.")

else:
    print("Could not find the best clustering result; export aborted. Please check the HPO process.")