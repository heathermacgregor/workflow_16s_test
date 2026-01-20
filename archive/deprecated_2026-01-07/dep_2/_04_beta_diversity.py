import logging
from pathlib import Path
from typing import Dict, List, Any
import anndata as ad
import pandas as pd
import skbio.diversity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from workflow_16s.logger import get_logger
from workflow_16s.visualization.beta_diversity import beta_diversity_plot

logger = get_logger()

class BetaDiversity:
    """Calculates beta diversity and stores ordination results in the AnnData object."""
    def __init__(self, config: Dict, adata: ad.AnnData):
        self.config = config
        self.adata = adata
        self.beta_config = self.config.get('beta_diversity', {})
        
        primary_colors = self.beta_config.get('color_columns', [])
        auto_colors = []
        if self.beta_config.get('analyze_all_valid_columns', True):
            auto_colors = self.adata.uns.get('analysis_columns', {}).get("group_comparison", [])
        
        all_colors = primary_colors + auto_colors
        self.color_columns = sorted(list(set(all_colors)))
        self.symbol_column = self.beta_config.get('symbol_column')

    def run(self, output_dir: Path) -> ad.AnnData:
        if not self.beta_config.get('enabled', True): return self.adata
        logger.info("STEP 4: Calculating beta diversity ordination...")
        for task in self._get_enabled_tasks():
            self._run_task(task, output_dir)
        return self.adata

    def _run_task(self, task: Dict, output_dir: Path):
        layer, method, task_name = task.get('layer'), task.get('method'), ""
        try:
            task_name = f"{method.upper()} on '{layer}' layer"
            if layer not in self.adata.layers:
                logger.warning(f"Beta diversity: Layer '{layer}' not found. Skipping task.")
                return
            
            data_matrix = self.adata.layers[layer].toarray() if hasattr(self.adata.layers[layer], 'toarray') else self.adata.layers[layer]
            obsm_key, uns_key, result_payload = f'X_{method}_{layer}', f'{method}_{layer}', {}

            if method == 'pca':
                pca = PCA(n_components=4)
                components_array = pca.fit_transform(data_matrix)
                components_df = pd.DataFrame(components_array, index=self.adata.obs_names, columns=[f'PC{i+1}' for i in range(components_array.shape[1])])
                self.adata.obsm[obsm_key] = components_array
                self.adata.uns[uns_key] = {'variance_ratio': pca.explained_variance_ratio_}
                result_payload = {'components': components_df, 'proportion_explained': pca.explained_variance_ratio_}
            
            elif method == 'pcoa':
                metric = task.get('metric', 'braycurtis')
                counts_matrix = self.adata.layers.get('counts')
                if counts_matrix is None: return logger.warning("PCoA requires 'counts' layer. Skipping.")
                counts_array = counts_matrix.toarray() if hasattr(counts_matrix, 'toarray') else counts_matrix
                dist_matrix = skbio.diversity.beta_diversity(metric, counts_array, self.adata.obs_names)
                pcoa_results = skbio.stats.ordination.pcoa(dist_matrix)
                
                pcoa_df = pcoa_results.samples
                pcoa_df.columns = [f'PCo{i+1}' for i in range(pcoa_df.shape[1])]
                
                obsm_key, uns_key = f'X_pcoa_{metric}', f'pcoa_{metric}'
                self.adata.obsm[obsm_key] = pcoa_df.values
                self.adata.uns[uns_key] = {'proportion_explained': pcoa_results.proportion_explained}
                result_payload = {'components': pcoa_df, 'proportion_explained': pcoa_results.proportion_explained}

            elif method in ['tsne', 'mds']:
                Model = TSNE if method == 'tsne' else MDS
                # Perplexity for t-SNE must be less than the number of samples
                perplexity = min(30, self.adata.n_obs - 1)
                model = Model(n_components=2, random_state=42, perplexity=perplexity) if method == 'tsne' else Model(n_components=2, random_state=42)
                components_array = model.fit_transform(data_matrix)
                components_df = pd.DataFrame(components_array, index=self.adata.obs_names, columns=['Dim1', 'Dim2'])
                self.adata.obsm[obsm_key] = components_array
                result_payload = {'components': components_df}

            elif method == 'umap':
                try:
                    from umap import UMAP
                    reducer = UMAP(n_components=2, random_state=42)
                    components_array = reducer.fit_transform(data_matrix)
                    components_df = pd.DataFrame(components_array, index=self.adata.obs_names, columns=['UMAP1', 'UMAP2'])
                    self.adata.obsm[obsm_key] = components_array
                    result_payload = {'components': components_df}
                except ImportError: return logger.error("UMAP requires 'umap-learn'. Please run 'pip install umap-learn'.")
            
            self._plot_ordination(result_payload, task, output_dir)
            logger.info(f"Successfully calculated {task_name}.")

        except Exception as e:
            logger.error(f"Beta diversity task {task_name} failed: {e}")

    def _get_enabled_tasks(self) -> List[Dict]:
        tasks = []
        for layer_name, layer_conf in self.beta_config.get('tables', {}).items():
            if layer_conf.get('enabled', False):
                for method in layer_conf.get('methods', []):
                    task = {'layer': layer_name, 'method': method}
                    if method == 'pcoa': task['metric'] = layer_conf.get('pcoa_metric', 'braycurtis')
                    tasks.append(task)
        return tasks

    def _plot_ordination(self, result: Dict, task: Dict, output_dir: Path):
        symbol_col = self.beta_config.get('symbol_column')
        for color_col in self.color_columns:
            if color_col not in self.adata.obs.columns: continue
            plot_dir = output_dir / "beta_diversity"; plot_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{task['method']}_{task['layer']}" + (f"_{task.get('metric')}" if task['method'] == 'pcoa' else "") + f"_{color_col}"
            beta_diversity_plot(components=result['components'], proportion_explained=result.get('proportion_explained'), metadata=self.adata.obs, color_col=color_col, symbol_col=symbol_col if symbol_col in self.adata.obs.columns else None, ordination_type=task['method'], transformation=task['layer'], output_path=plot_dir / filename)