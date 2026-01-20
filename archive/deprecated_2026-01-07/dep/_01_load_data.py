# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-Party Imports
import numpy as np
import pandas as pd
from biom.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

# Local Imports from workflow_16s
from workflow_16s.constants import TAXONOMIC_LEVELS, TAXONOMY_PREFIXES
from workflow_16s.downstream import Data
from workflow_16s.downstream._env import env
from workflow_16s.downstream._nfc import update_nfc_facilities_data
from workflow_16s.logger import get_logger
from workflow_16s.utils.biom_utils import import_merged_feature_table
from workflow_16s.utils.data import sync_samples
from workflow_16s.utils.metadata_utils import (
    clean_metadata, import_merged_metadata, SampleProcessor
)

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = get_logger()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# ============================ DASHBOARD CLASS ============================ #

class MetadataDashboard:
    """
    An interactive Dash application with a vertical layout and fixed-height plots.
    """
    def __init__(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame, figures: Dict[str, go.Figure], analysis_columns: Dict[str, List[str]]):
        self.orig_df = original_df
        self.df = filtered_df
        self.figures = figures
        self.analysis_columns = analysis_columns
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Metadata Summary Dashboard"
        
        # Use the pre-categorized lists to populate the dropdowns,
        # ensuring only relevant columns are shown.
        self.numerical_cols = self.analysis_columns.get("correlation_gradient", [])
        self.categorical_cols = self.analysis_columns.get("group_comparison", [])
        
        self.dataset_id_col = next((c for c in ['dataset_name', 'dataset_id'] if c in self.df.columns), None)

        self._create_layout()
        self._register_callbacks()

    def _create_layout(self):
        """Builds the Dash application layout with vertical stacking."""
        self.app.layout = dbc.Container([
            html.H1("🔬 Metadata Summary Dashboard", className="my-4 text-center"),
            # Summary Cards stacked vertically
            dbc.Row([
                dbc.Col([
                    self._create_summary_card("Total Samples", self.df.shape[0]),
                    self._create_summary_card("Unique Datasets", self.df[self.dataset_id_col].nunique() if self.dataset_id_col else "N/A"),
                    self._create_summary_card("Columns (Before Filter)", self.orig_df.shape[1]),
                    self._create_summary_card("Columns (After Filter)", self.df.shape[1]),
                ], width=12, lg=6, className="mx-auto") # Center the column on large screens
            ], className="mb-4 justify-content-center"),
            
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="overview", children=self._create_overview_tab()),
                dbc.Tab(label="Variable Explorer", tab_id="explorer", children=self._create_explorer_tab()),
            ])
        ], fluid=True)

    def _create_summary_card(self, title, value):
        # Cards now have a bottom margin for spacing
        return dbc.Card(dbc.CardBody([
            html.H4(title, className="card-title"),
            html.P(f"{value}", className="card-text fs-2 fw-bold"),
        ]), color="light", className="mb-3")
    
    def _create_overview_tab(self):
        """Layout for the overview tab with fixed-height, vertically stacked plots."""
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="completeness-plot", figure=self.figures.get('completeness'), style={'height': '600px'}),
                html.Hr(), # Separator line
                dcc.Graph(id="sample-dist-plot", figure=self.figures.get('sample_distribution'), style={'height': '600px'}),
            ], width=12)
        ], className="mt-4")

    def _create_explorer_tab(self):
        """Layout for the variable explorer with fixed-height, vertically stacked plots."""
        return html.Div([
            dbc.Row([
                dbc.Col(html.Label("Select Numerical Column:"), width=12),
                dbc.Col(dcc.Dropdown(id='numerical-dropdown', options=self.numerical_cols, value=self.numerical_cols[0] if self.numerical_cols else None), width=12, lg=6),
            ], className="my-4"),
            dcc.Graph(id='numerical-plot', style={'height': '500px'}),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col(html.Label("Select Categorical Column:"), width=12),
                dbc.Col(dcc.Dropdown(id='categorical-dropdown', options=self.categorical_cols, value=self.categorical_cols[0] if self.categorical_cols else None), width=12, lg=6),
            ], className="my-4"),
            dcc.Graph(id='categorical-plot', style={'height': '500px'}),
            dbc.RadioItems(id='categorical-plot-type', options=[{'label': 'Bar', 'value': 'bar'}, {'label': 'Pie', 'value': 'pie'}], value='bar', inline=True, className="d-flex justify-content-center mt-2 p-3")
        ])

    def _register_callbacks(self):
        """Registers callbacks for the dynamic 'Variable Explorer' tab."""
        @self.app.callback(Output('numerical-plot', 'figure'), Input('numerical-dropdown', 'value'))
        def update_numerical_plot(selected_col):
            if not selected_col: return go.Figure(layout={"title": "Select a numerical column", "height": 500})
            return px.histogram(self.df, x=selected_col, title=f'Distribution of {selected_col}', height=500)

        @self.app.callback(Output('categorical-plot', 'figure'), [Input('categorical-dropdown', 'value'), Input('categorical-plot-type', 'value')])
        def update_categorical_plot(selected_col, plot_type):
            if not selected_col: return go.Figure(layout={"title": "Select a categorical column", "height": 500})
            counts = self.df[selected_col].value_counts().nlargest(25)
            if plot_type == 'bar':
                fig = px.bar(counts, y=counts.index, x=counts.values, orientation='h', title=f'Top 25 Categories for {selected_col}')
            else:
                fig = px.pie(names=counts.index, values=counts.values, title=f'Top 25 Categories for {selected_col}')
            fig.update_layout(height=500) # Ensure fixed height for both plot types
            return fig

    def run(self):
        """Starts the Dash server."""
        self.app.run(debug=True, host='0.0.0.0')

# ================================= DATA LOADING ===================================== #

class DataLoader:
    """
    Loads and aligns BIOM feature tables and a unified metadata file using
    a targeted, table-driven approach for path discovery.
    """
    ModeConfig = {
        "asv": ("asv", "table", "asv"),
        "genus": ("genus", "table_6", "l6"),
    }

    def __init__(self, config: Dict, existing_subsets: Any = None):
        self.config = config
        self.project_dir = Path(config.get("project_dir", "."))
        self.verbose = config.get("verbose", False)
        self.existing_subsets = existing_subsets
        self.data = Data()
        self.original_metadata: Optional[pd.DataFrame] = None
        # Directory for saving visual outputs
        self.results_dir = self.project_dir / "results" / "metadata_plots"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures: Dict[str, go.Figure] = {}

    def run(self) -> Data:
        """Executes the data loading and alignment process."""
        # 1. Load and merge all metadata using the targeted discovery method
        self.data.metadata = self._load_unified_metadata()

        # 2. Load feature tables and align them with the unified metadata
        if self.config.get("target_subfragment_mode", "genus") != "any":
            self._load_and_align_table("asv")
        self._load_and_align_table("genus")

        # 3. Load supplementary data
        if "genus" in self.data.tables.get("raw", {}):
            if self.config.get("nfc_facilities", {}).get("enabled", False):
                self._process_nfc_data()
            if self.config.get("environmental_data", {}).get("enabled", False):
                self._process_env_data()
        
        # 4. Summarize, filter, and report on the final metadata
        self._summarize_and_filter_metadata()
        
        self.data.analysis_columns = self._categorize_columns_for_analysis()
        
        if self.data.metadata is not None:
            # Define the output path
            output_path = self.project_dir / "data" / "merged" / "metadata" / "final_metadata.tsv"
            
            # Save the DataFrame to a TSV file
            self.data.metadata.to_csv(output_path, sep='\t', index=False)
            logger.info(f"✅ Filtered metadata DataFrame saved to: {output_path}")

        return self.data
    
    def launch_dashboard(self):
        """Initializes and runs the interactive Dash dashboard if enabled in the config."""
        if not self.config.get("dashboard", {}).get("enabled", False):
            logger.info("Dashboard is disabled in the configuration. Skipping launch.")
            return
            
        if self.original_metadata is None or self.data.metadata is None:
            logger.error("Metadata has not been loaded. Run the .run() method first.")
            return

        if 'completeness' not in self.figures or 'sample_distribution' not in self.figures:
            logger.warning("Dashboard enabled, but initial figures were not generated.")
            self.figures.setdefault('completeness', go.Figure())
            self.figures.setdefault('sample_distribution', go.Figure())

        logger.info("Launching metadata dashboard...")
        dashboard = MetadataDashboard(
            original_df=self.original_metadata,
            filtered_df=self.data.metadata,
            figures=self.figures,
            analysis_columns=self.data.analysis_columns  
        )
        dashboard.run()

    def _load_unified_metadata(self) -> pd.DataFrame:
        """
        Loads a single, unified metadata DataFrame by first finding all BIOM
        tables and then deriving their corresponding metadata paths.
        """
        logger.info("Discovering metadata files based on found BIOM tables...")
        all_table_paths = []

        # Step 1: Find all possible BIOM table paths for all modes
        if self.config.get("target_subfragment_mode", "genus") != "any":
            all_table_paths.extend(self._get_table_paths('asv', self.ModeConfig['asv'][1]))
        all_table_paths.extend(self._get_table_paths('genus', self.ModeConfig['genus'][1]))

        if not all_table_paths:
            raise FileNotFoundError("No BIOM tables were found, cannot discover metadata.")

        # Step 2: Derive metadata paths from the list of table paths
        metadata_paths = self._get_metadata_paths_from_tables(all_table_paths)

        if not metadata_paths:
            raise FileNotFoundError("Could not derive any metadata paths from the found BIOM tables.")

        # Step 3: De-duplicate the list of derived paths and load the data
        unique_metadata_paths = list(set(metadata_paths))

        metadata = import_merged_metadata(unique_metadata_paths) # type: ignore
        metadata = clean_metadata(self.config, metadata)
        #metadata = SampleProcessor(metadata).process()
        #metadata['identified_latitude'] = metadata.apply(find_latitude, axis=1)
        logger.info(metadata.dropna().head(5).to_string(index=False))
        logger.info(f"Derived {len(metadata_paths)} metadata paths, loading {len(unique_metadata_paths)} unique files.")
        logger.info(f"Loaded unified metadata: {metadata.shape[0]} samples × {metadata.shape[1]} columns")
        return metadata
        
    def _get_metadata_paths_from_tables(self, table_paths: List[Path]) -> List[Path]:
        """
        Derives metadata file paths from a list of BIOM table paths.
        This is the "old method" of targeted path discovery.
        """
        if self.existing_subsets is not None:
            return [Path(paths["metadata"]) for _, paths in self.existing_subsets.items()]

        tsv_paths: List[Path] = []
        if self.project_dir is None:
            raise ValueError("project_dir must be set in the config before deriving metadata paths.")
        metadata_dir = Path(self.project_dir) / "data" / "per_dataset" / "metadata"
        
        for table_path in table_paths:
            try:
                # This logic assumes a parallel directory structure.
                # It takes the key subdirectories from the BIOM path to build the metadata path.
                tail = table_path.parts[-7:-2]
                tsv_path = metadata_dir.joinpath(*tail, "sample-metadata.tsv")
                if tsv_path.exists():
                    tsv_paths.append(tsv_path)
                else:
                    logger.warning(f"Metadata file not found for table {table_path} at expected path {tsv_path}")
            except IndexError:
                logger.warning(f"Could not derive metadata path from oddly structured table path: {table_path}")
        return tsv_paths

    def _load_and_align_table(self, mode: str) -> None:
        """Loads a feature table and aligns it with the master metadata."""
        level, subdir, _ = self.ModeConfig[mode]
        table = self._load_biom_table(level, subdir)
        if table is not None and hasattr(table, "shape") and table.shape is not None:
            logger.info(f"Table loaded for level '{level}': {table.shape[0]} features × {table.shape[1]} samples")
        else:
            logger.warning(f"Table for level '{level}' is None or missing shape attribute.")
        if self.data.metadata is not None:
            table, self.data.metadata = sync_samples(table, self.data.metadata)
        else:
            logger.error("Metadata is None, cannot align samples with feature table.")
        self._log_results(table, self.data.metadata, level)
        self.data.tables["raw"][level] = table

    def _load_biom_table(self, level: str, subdir: str) -> Table:
        """Loads and merges BIOM tables from discovered file paths."""
        table_paths = self._get_table_paths(level, subdir)
        if not table_paths:
            raise FileNotFoundError(f"No BIOM table files found for level '{level}'")
        # Convert Path objects to str for compatibility if needed
        biom_paths = [str(p) if isinstance(p, Path) else p for p in table_paths]
        return import_merged_feature_table(biom_paths) # type: ignore TODO

    def _get_table_paths(self, level: str, subdir: str) -> List[Path]:
        """Discovers BIOM table file paths based on configuration."""
        if self.existing_subsets is not None:
            paths_str = [p[subdir] for _, p in self.existing_subsets.items()]
        else:
            subfrag = "*" if self.config.get("target_subfragment_mode") == "any" else self.config.get("target_subfragment_mode")
            if not self.project_dir:
                raise ValueError("project_dir must be set in the config.")
            if not subfrag:
                raise ValueError("target_subfragment_mode must be set in the config or default to '*'.")
            pattern = str(Path(str(self.project_dir)) / "data" / "per_dataset" / "qiime" / "*" / "*" / "*" / str(subfrag) / "FWD_*_REV_*" / subdir / "feature-table.biom")
            paths_str = glob.glob(pattern, recursive=True)
        
        paths = [Path(p) for p in paths_str]
        # This log is now less critical for metadata but still useful for tables
        if self.verbose:
            logger.info(f"Found {len(paths)} feature tables for level '{level}'")
        return paths
        
    def _process_nfc_data(self):
        """Loads and processes Nuclear Fuel Cycle (NFC) facilities data."""
        logger.info("Processing NFC facilities data...")
        try:
            if self.data.metadata is None:
                logger.error("Cannot process NFC data: metadata is None.")
                return
            nfc, updated_metadata = update_nfc_facilities_data(self.config, self.data.metadata)
            self.data.nfc_facilities = nfc
            self.data.metadata = updated_metadata
        except Exception as e:
            logger.error(f"Failed to process NFC data: {e}\n{traceback.format_exc()}")

    def _process_env_data(self):
        """Loads environmental data from external sources."""
        logger.info("Loading environmental data...")
        try:
            if self.data.metadata is None:
                logger.error("Cannot process environmental data: metadata is None.")
                return
            table = self.data.tables["raw"]["genus"]
            self.data.metadata = env(self.config, table, self.data.metadata)
        except Exception as e:
            logger.error(f"Failed to load environmental data: {e}\n{traceback.format_exc()}")
            
            
    def _categorize_columns_for_analysis(self) -> Dict[str, List[str]]:
        """
        Automatically categorizes metadata columns for different downstream analyses.
        
        Returns:
            A dictionary containing lists of column names for each category.
        """
        if self.data.metadata is None:
            logger.warning("Metadata not loaded, cannot categorize columns.")
            return {}

        df = self.data.metadata
        analysis_cols = {
            "group_comparison": [],
            "correlation_gradient": [],
            "potential_confounders": []
        }

        # Define columns that are typically identifiers or high-cardinality
        # and should be excluded from most analyses.
        id_like_cols = {'sampleid', '#sampleid', 'sample_id', 'barcode', 'description'}
        lat_lon_cols = {'latitude', 'longitude', 'lat', 'lon', 'latitude_deg', 'longitude_deg'}
        
        # Define columns that often represent technical batches
        confounder_cols = {'sequencing_run', 'run_id', 'batch_number', 'dna_extraction_kit'}

        for col in df.columns:
            if col in id_like_cols or col in lat_lon_cols:
                continue # Skip common ID columns

            # Check for potential confounders first
            if col in confounder_cols:
                analysis_cols["potential_confounders"].append(col)
                continue

            # Categorize as continuous if it's a numeric type
            if pd.api.types.is_numeric_dtype(df[col]):
                # Ensure it's not just a binary (0/1) or low-variety integer column
                if df[col].nunique() > 10:
                    analysis_cols["correlation_gradient"].append(col)
                else:
                    # Treat low-variety numeric cols as potential grouping variables
                    analysis_cols["group_comparison"].append(col)
            
            # Categorize as categorical if it's an object/category type
            elif pd.api.types.is_object_dtype(df[col]) or df[col].dtype.name == 'category':
                # Only include low-cardinality categoricals for grouping.
                # Heuristic: A column is "low-cardinality" if its unique values
                # are less than 50% of the total number of samples and also less than 50 total.
                if df[col].nunique() < min(50, len(df) * 0.5):
                     analysis_cols["group_comparison"].append(col)

        logger.info("Successfully categorized metadata columns for analysis.")
        return analysis_cols
            
    def _generate_visual_summaries(self, df: pd.DataFrame, dataset_id_col: Optional[str], numerical_cols: List[str], categorical_cols: List[str], subfolder: str):
        plot_dir = self.results_dir / subfolder
        plot_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Checking for visual generation tasks in: {plot_dir}")

        if self.config.get("dashboard", {}).get("enabled", False):
            logger.info("Initializing/updating dashboard components...")
            if dataset_id_col:
                top_datasets = df[dataset_id_col].value_counts().nlargest(25)
                fig_samples = px.bar(
                    top_datasets, y=top_datasets.index, x=top_datasets.values, orientation='h',
                    title=f'Sample Distribution (Top 25) - {subfolder.replace("_", " ").title()}'
                ).update_layout(yaxis={'categoryorder':'total ascending'})
                self.figures['sample_distribution'] = fig_samples
            
            # The completeness plot should only be generated BEFORE filtering to be most informative.
            if subfolder == "before_filtering":
                completeness = (df.notnull().sum() / len(df) * 100).sort_values(ascending=True)
                fig_completeness = px.bar(
                    completeness, y=completeness.index, x=completeness.values, orientation='h',
                    title='Metadata Column Completeness (Before Filtering)',
                    height=max(400, len(df.columns) * 20)
                ).update_layout(yaxis={'categoryorder':'total ascending'})
                self.figures['completeness'] = fig_completeness

        if not self.config.get("visuals", {}).get("enabled", True):
            logger.info("Static visual generation is disabled in config.")
            return
        
        pie_plot_dir = plot_dir / "categorical_pie_charts"
        pie_plot_dir.mkdir(exist_ok=True)
        for col in list(set(categorical_cols + ['target_gene', 'target_subfragment'])):
            if col in df.columns and 1 < df[col].nunique() < 50:
                try:
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, 'count']
                    fig = px.pie(counts, names=col, values='count', title=f'Distribution of {col} ({subfolder})')
                    pio.write_html(fig, str(pie_plot_dir / f"{col}_pie_chart.html"))
                except Exception as e:
                    logger.warning(f"Could not generate pie chart for '{col}': {e}")


    def _summarize_and_filter_metadata(self):
        """
        Summarizes, filters, reports on, and visualizes the metadata DataFrame.
        
        This function performs an initial summary, filters out sparse columns
        based on a threshold, and then provides a detailed summary of the
        remaining columns. Visualizations are generated before and after filtering.
        The instance's metadata object is updated in place. All logged output
        from this function is also saved to 'results/metadata_summary.log'.
        """
        if self.data.metadata is None:
            logger.error("Metadata is None, cannot perform summary and filtering.")
            return

        # Create an empty file and get a file object
        log_file = self.results_dir.parent / "metadata_summary.log"
        with open(log_file, 'w') as f:
            def write_to_file(message):
                f.write(message + '\n')
            write_to_file("--- Starting Metadata Summary, Filtering, and Visualization ---")
            try:
                threshold = self.config.get("metadata_completeness_threshold", 25) / 100.0
                write_to_file(f"Using completeness threshold: {threshold*100:.0f}%")
                df = self.data.metadata
                self.original_metadata = df.copy()

                # Helper function to perform summary tasks, avoiding code repetition
                def _perform_summary(df: pd.DataFrame, summary_title: str):
                    write_to_file(f"--- {summary_title} ---")
                    
                    # 1. Find Sample ID column and get total samples
                    sample_id_col = None
                    for col in ['#sampleid', 'sample_id', 'sampleid', 'sample-id']:
                        if col in df.columns:
                            sample_id_col = col
                            break
                    
                    if sample_id_col:
                        write_to_file(f"Using '{sample_id_col}' as the sample identifier.")
                        df = df.set_index(sample_id_col, drop=False)
                        write_to_file(f"Total number of samples: {df.shape[0]}")
                    else:
                        write_to_file(f"WARNING: Could not find a standard sample ID column. Found: {list(df.columns)}")
                        write_to_file(f"Total number of samples: {df.shape[0]}")

                    # 2. Find Dataset ID column and count unique datasets
                    dataset_id_col = None
                    for col in ['dataset_name', 'dataset_id']:
                        if col in df.columns:
                            dataset_id_col = col
                            break
                    
                    if dataset_id_col:
                        write_to_file(f"Using '{dataset_id_col}' as the dataset identifier.")
                        write_to_file(f"Found {df[dataset_id_col].nunique()} unique datasets.")
                        
                        # 3. Summarize how many samples each dataset has
                        write_to_file("Sample distribution per dataset:")
                        for dataset, count in df.groupby(dataset_id_col).size().items():
                            write_to_file(f"  - {dataset}: {count} samples")
                    else:
                        write_to_file("WARNING: Could not find a dataset identifier column.")

                    # 4. Summarize information about key columns
                    key_cols = [
                        'target_subfragment', 'target_gene', 'pcr_primer_fwd', 
                        'pcr_primer_rev', 'pcr_primer_fwd_seq', 'pcr_primer_rev_seq', 
                        'instrument_platform', 'instrument_model', 'library_layout'
                    ]
                    write_to_file("Summary of key columns:")
                    for col in key_cols:
                        if col in df.columns:
                            completeness = (df[col].count() / len(df)) * 100
                            write_to_file(f"  - '{col}': Overall Completeness = {completeness:.2f}%, Unique Values = {df[col].nunique()}")
                            if dataset_id_col:
                                write_to_file("    Completeness per dataset:")
                                for dataset, perc in df.groupby(dataset_id_col)[col].apply(lambda x: x.count() / len(x) * 100).items():
                                    write_to_file(f"      - {dataset}: {perc:.2f}%")
                        else:
                            write_to_file(f"  - Column '{col}' not found.")

                # --- Identify Column Types ---
                categorical_cols, binary_cols, numerical_cols = [], [], []
                for col in df.columns:
                    # Ensure you're working with a string for the column name
                    # This is a safeguard in case the loop variable is a list
                    column_name = col[0] if isinstance(col, list) else col

                    if pd.api.types.is_numeric_dtype(df[column_name]):
                        numerical_cols.append(column_name)
                    else:
                        categorical_cols.append(column_name)
                    """
                    elif df[column_name].dtypes in ['object', 'category', 'bool']:
                        if df[column_name].nunique(dropna=False) == 2:
                            binary_cols.append(column_name)
                        else:
                            categorical_cols.append(column_name)
                    """

                # --- Initial Summary & Visualization ---
                _perform_summary(df, "Initial Metadata Summary (Before Filtering)")
                dataset_id_col = next((c for c in ['dataset_name', 'dataset_id'] if c in df.columns), None)
                """
                self._generate_visual_summaries(
                    df=df,
                    dataset_id_col=dataset_id_col,
                    numerical_cols=numerical_cols,
                    categorical_cols=categorical_cols + binary_cols,
                    subfolder="before_filtering"
                )
                """
                
                # --- Column Sorting and Filtering ---
                write_to_file("\n--- Sorting and Filtering Columns by Completeness ---")
                min_samples_required = int(threshold * len(df))
                # 1. Get the counts of non-missing values for every column
                counts = df.count()

                # 2. Filter that Series to find which columns meet the condition
                # 3. Get the names (the index) of those columns and convert to a list
                cols_to_drop = counts[counts < min_samples_required].index.tolist()
                
                if cols_to_drop:
                    write_to_file(f"Dropping {len(cols_to_drop)} columns failing to meet {threshold*100:.0f}% completeness threshold ({min_samples_required} samples):")
                    for col in cols_to_drop:
                        write_to_file(f"  - '{col}' (Completeness: {(df[col].count() / len(df)) * 100:.2f}%)")
                    #self.data.metadata.drop(columns=cols_to_drop, inplace=True) # TODO: COME BACK LATER
                    write_to_file(f"Metadata shape after filtering: {self.data.metadata.shape}")
                else:
                    write_to_file("No columns were dropped based on the completeness threshold.")
                    
                def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
                    """Makes DataFrame column names unique by appending suffixes if needed."""
                    cols = pd.Series(df.columns)
                    for dup in cols[cols.duplicated()].unique():
                        cols[cols[cols == dup].index.values.tolist()] = [
                            f"{dup}_{i}" for i in range(sum(cols == dup))
                        ]
                    df.columns = cols
                    return df
                
                # Store filtered DataFrame in a local variable
                df_filtered = self.data.metadata
                
                # Ensure columns are unique to avoid confusion in summaries
                df_filtered = make_columns_unique(df_filtered)
                
                # --- Detailed Summary of Remaining Columns ---
                write_to_file("\n--- Detailed Summary of Remaining Columns (After Filtering) ---")
                remaining_cat_bin = [c for c in categorical_cols + binary_cols if c in df_filtered.columns]
                if remaining_cat_bin:
                    write_to_file("\nSummary of Categorical and Binary Columns:")
                    for col in remaining_cat_bin:
                        # Calculate summary statistics for the current column 'col'
                        total_rows = len(df_filtered)
                        if total_rows > 0:
                            empty_pct = (df_filtered[col].isnull().sum() / total_rows) * 100
                        else:
                            empty_pct = 0
                        
                        write_to_file(f"\nColumn: '{col}' | Unique Values: {df_filtered[col].nunique()} | Empty Cells: {empty_pct:.2f}%")
                        write_to_file("  Composition (Top 5):")
                        for value, percentage in df_filtered[col].value_counts(normalize=True).head(5).items():
                            write_to_file(f"    - {str(value)[:60]:<60}: {percentage:.2%}")

                remaining_num = [c for c in numerical_cols if c in df_filtered.columns]
                if remaining_num:
                    write_to_file("\nSummary of Numerical Columns:")
                    write_to_file(df_filtered[remaining_num].describe().transpose().to_string())
                    write_to_file("\nEmpty Cell Percentage for Numerical Columns:")
                    for col in remaining_num:
                        empty_pct = df_filtered[col].isnull().sum() / len(df_filtered) * 100
                        write_to_file(f"  - '{col}': {empty_pct:.2f}%")
                        
                # --- Final Summary & Visualization ---
                write_to_file("\n" + "="*50)
                _perform_summary(df_filtered, "Final Metadata Summary (After Filtering)")
                self._generate_visual_summaries(
                    df=df_filtered,
                    dataset_id_col=dataset_id_col,
                    numerical_cols=[c for c in numerical_cols if c in df_filtered.columns],
                    categorical_cols=[c for c in categorical_cols + binary_cols if c in df_filtered.columns],
                    subfolder="after_filtering"
                )
                write_to_file("--- Metadata Summary, Filtering, and Visualization Complete ---")

            finally:
                # Log a message to the console indicating the file has been written.
                logger.info(f"Metadata summary report saved to: {log_file}")

    def _log_results(self, table: Table, metadata: Optional[pd.DataFrame], level: str) -> None:
        """Logs summary statistics for the loaded data."""
        if table.is_empty() or table.shape is None:
            shape = "Empty"
        else:
            shape = f"{table.shape[0]} features × {table.shape[1]} samples"
        logger.info(f"Loaded and aligned '{level}' features: {shape}")



if __name__ == "__main__":
    from workflow_16s.config import get_config # type: ignore
    config = get_config()
    
    project_dir = Path(config.get("project_dir", "."))
    results_dir = project_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # --- Step 1: Load Data ---
    logger.info("STEP 1: Loading and aligning data...")
    loader = DataLoader(config)
    data_object = loader.run()
    # --- Step 2: Launch the dashboard using the integrated method ---
    loader.launch_dashboard()
    logger.info("STEP 1: Data loading complete.")