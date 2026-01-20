# ===================================== IMPORTS ====================================== #

import logging
from pathlib import Path
from typing import Dict, Union
import anndata as ad
import pandas as pd
import numpy as np
from workflow_16s.config import get_config
from workflow_16s.logger import get_logger
from ._01_load_data import DataLoader
from ._02_transform_tables import DataProcessor
from ._03_alpha_diversity import AlphaDiversity
from ._04_beta_diversity import BetaDiversity
from ._05_statistical_tests_a import StatisticalTests
from ._06_statistical_tests_b import AdvancedAnalyses
from ._07_feature_selection import FeatureSelection
from ._08_comprehensive_analysis import ComprehensiveAnalysis

logger = get_logger()

# ============================ CORE PIPELINE FUNCTIONS =============================== #

def run_analysis_pipeline(config: Dict, adata: ad.AnnData, output_dir: Path) -> ad.AnnData:
    adata = DataProcessor(config, adata).run()
    adata = AlphaDiversity(config, adata).run(output_dir)
    adata = BetaDiversity(config, adata).run(output_dir)
    adata = StatisticalTests(config, adata).run(output_dir)
    adata = AdvancedAnalyses(config, adata).run(output_dir)
    adata = FeatureSelection(config, adata).run(output_dir)
    ComprehensiveAnalysis(config, adata).run(output_dir)
    return adata

def run_full_workflow(config: Dict) -> Union[ad.AnnData, Dict[str, ad.AnnData]]:
    project_dir = Path(config.get("project_dir", "."))
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)
    strategy = config.get("analysis_strategy", "combined")
    loader = DataLoader(config)
    if strategy == "combined":
        logger.info("--- Executing COMBINED analysis strategy ---")
        merged_adata = loader.run_combined()
        if merged_adata is None or merged_adata.n_obs == 0: return {}
        return run_analysis_pipeline(config, merged_adata, results_dir)
    elif strategy == "separate":
        logger.info("--- Executing SEPARATE analysis strategy ---")
        all_results: Dict[str, ad.AnnData] = {}
        datasets = loader.discover_datasets()
        for i, (dataset_id, path) in enumerate(datasets.items()):
            logger.info(f"\n{'='*20} PROCESSING DATASET {i+1}/{len(datasets)}: {dataset_id} {'='*20}")
            dataset_results_dir = results_dir / dataset_id
            dataset_results_dir.mkdir(exist_ok=True)
            single_adata = loader.run_single(path)
            if single_adata is None: continue
            all_results[dataset_id] = run_analysis_pipeline(config, single_adata, dataset_results_dir)
        return all_results
    else:
        raise ValueError(f"Unknown analysis_strategy '{strategy}'. Must be 'separate' or 'combined'.")

# ============================ EXAMPLE DATA GENERATOR ============================== #

def create_dummy_anndata(n_obs: int = 100, n_vars: int = 500) -> ad.AnnData:
    logger.info(f"Generating a dummy AnnData object with {n_obs} samples and {n_vars} features...")
    counts = np.random.randint(0, 2000, size=(n_obs, n_vars))
    obs_df = pd.DataFrame({
        'group': np.random.choice(['Control', 'Treatment'], size=n_obs),
        'site': np.random.choice(['Gut', 'Skin', 'Oral'], size=n_obs, p=[0.6, 0.3, 0.1]),
        'ph': np.random.normal(7.0, 0.5, size=n_obs),
        'patient_id': [f'Patient_{i%10}' for i in range(n_obs)],
        'nuclear_contamination_status': np.random.choice(['Contaminated', 'Not Contaminated'], size=n_obs, p=[0.1, 0.9]),
    }, index=[f'Sample_{i}' for i in range(n_obs)])

    # Inject a simple signal for the ML model to find
    treatment_indices = obs_df['group'] == 'Treatment'
    counts[treatment_indices, :10] += np.random.randint(500, 1000, size=(treatment_indices.sum(), 10))
    
    var_df = pd.DataFrame({'Taxonomy': [f'k__B;p__F;c__C;o__C;f__L;g__G{i%10}' if i % 3 == 0 else f'k__B;p__B;c__B;o__B;f__B;g__B' for i in range(n_vars)]}, index=[f'ASV_{i}' for i in range(n_vars)])
    adata = ad.AnnData(X=counts, obs=obs_df, var=var_df)
    adata.layers['counts'] = adata.X.copy()
    return adata

# ============================ STANDALONE EXECUTION BLOCK ========================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 16S rRNA upstream analysis pipeline.")
    parser.add_argument("-c", "--config", type=Path, 
                        default=Path("/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml"), 
                        help="Path to the YAML configuration file for the workflow.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = get_logger()
    
    config_path = args.config
    try:
        with open(config_path, 'r') as f: config_dict = yaml.safe_load(f)
        config = AppConfig(**config_dict)
        
        log_level_str = getattr(config, 'logging_level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
        logger.info(f"Logger level set to {log_level_str}.")

    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'")
        exit(1)
    except ValidationError as e:
        logger.error(f"Configuration file is invalid:\n{e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        exit(1)

    datasets_processor = Upstream(config)
    if datasets_processor: asyncio.run(datasets_processor.execute())

'''
if __name__ == "__main__":
    logger.info("--- RUNNING DOWNSTREAM WORKFLOW WITH EXAMPLE DATA ---")
    example_output_dir = Path("./downstream_example_results")
    example_output_dir.mkdir(exist_ok=True)
    example_adata = create_dummy_anndata()
    example_config = {
        'project_dir': str(example_output_dir), 'threads': 4,
        'features': {'filter': {'enabled': True}, 'normalize': {'enabled': True}, 'clr_transform': {'enabled': True}, 'presence_absence': {'enabled': True}},
        'alpha_diversity': {'enabled': True, 'metrics': ['shannon', 'sobs'], 'correlation_analysis': {'enabled': True}},
        'beta_diversity': {
            'enabled': True, 'color_columns': ['group', 'site'], 'symbol_column': 'nuclear_contamination_status',
            'tables': {
                'clr': {'enabled': True, 'methods': ['pca']},
                'normalized': {'enabled': True, 'methods': ['pcoa'], 'pcoa_metric': 'braycurtis'}
            }
        },
        'stats': {'enabled': True, 'group_columns': ['group', 'site'], 'tables': {'filtered': {'enabled': True, 'tests': ['kruskal_bonferroni']}, 'presence_absence': {'enabled': True, 'tests': ['fisher']}}},
        'advanced_analyses': {'enabled': True, 'core_microbiome': {'enabled': True, 'group_columns': ['site']}},
        'ml': {'enabled': True, 'target_columns': ['group'], 'tables': {'clr': {'enabled': True, 'levels': ['asv'], 'methods': ['rfe']}}},
        'comprehensive_analysis': {'enabled': True}
    }
    run_analysis_pipeline(config=example_config, adata=example_adata, output_dir=example_output_dir)
    logger.info(f"✅ Example run finished successfully! Results are in '{example_output_dir}'.")
'''