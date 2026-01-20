# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Imports
import pandas as pd
from scipy.spatial.distance import cdist

# Local imports
import workflow_16s.custom_tmp_config  
import workflow_16s.sequences.analyze as seq_analyze
from workflow_16s import constants
from workflow_16s.ena.metadata import ENAMetadata
from workflow_16s.utils import dir_utils, file_utils, misc_utils

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def parse_sample_pooling(df: pd.DataFrame) -> pd.DataFrame:
    """Parses pooled samples in the 'sample_description' column into separate columns.
    
    Args:
        df:

    Returns:
        combined_df:
    """

    def _parse_sample_pooling(row) -> pd.DataFrame:
        sample_description = row.get("sample_description", "")
        data_str = sample_description.strip("''").replace(
            "Samples were pooled in this way (MID LONGITUDE LATITUDE SAMPLING_DATE ELEVATION) : ",
            "",
        )
        data_list = data_str.split()
        num_columns = 5
        data_reshaped = [
            data_list[i : i + num_columns]
            for i in range(0, len(data_list), num_columns)
        ]
        columns = [
            "MID",
            "LONGITUDE",
            "LATITUDE",
            "SAMPLING_DATE",
            "ELEVATION",
        ]
        parsed_df = pd.DataFrame(data_reshaped, columns=columns)
        for col in row.index:
            parsed_df[col] = row[col]
        return parsed_df

    parsed_dfs = df.apply(_parse_sample_pooling, axis=1)
    combined_df = (
        pd.concat(parsed_dfs.values, ignore_index=True)
        .rename(
            columns={
                "MID": "barcode_sequence",
                "LONGITUDE": "longitude_deg",
                "LATITUDE": "latitude_deg",
                "SAMPLING_DATE": "sampling_date",
                "ELEVATION": "elevation_m",
            }
        )
        .reset_index(drop=True)
    )
    return combined_df


def calculate_distances(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> pd.DataFrame:
    """Calculate Euclidean distances between two DataFrames based on 
    longitude and latitude.
    """
    df1[["longitude_deg", "latitude_deg"]] = df1[
        ["longitude_deg", "latitude_deg"]
    ].apply(pd.to_numeric)

    df2[["longitude_deg", "latitude_deg"]] = df2[
        ["longitude_deg", "latitude_deg"]
    ].apply(pd.to_numeric)

    distances = cdist(
        df1[["longitude_deg", "latitude_deg"]],
        df2[["longitude_deg", "latitude_deg"]],
        metric="euclidean",
    )
    min_dist_indices = distances.argmin(axis=1)
    return df2.iloc[min_dist_indices].reset_index(drop=True)


# ========================== CORE PROCESSING CLASS ========================== #

class SubsetDataset:
    """Central processing unit for dataset analysis with automated and manual
    modes.

    Features:
        - Automated primer estimation and metadata validation.
        - Manual metadata and primer configuration.
        - Error tracking and success/failure reporting.

    Attributes:
        config:  Configuration dictionary for processing parameters.
        dirs:    Subdirectories structure handler.
        success: List of successfully processed datasets with parameters.
        failed:  List of failed datasets with error information.
    """

    ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)
    ENA_METADATA_UNNECESSARY_COLUMNS = [
        "sra_bytes",
        "sra_aspera",
        "sra_galaxy",
        "sra_md5",
        "sra_ftp",
        "fastq_bytes",
        "fastq_aspera",
        "fastq_galaxy",
        "fastq_md5",
        "collection_date_start",
        "collection_date_end",
        "location_start",
        "location_end",
        "ncbi_reporting_standard",
        "datahub",
        "tax_lineage",
        "tax_id",
        "scientific_name",
        "isolation_source",
        "first_created",
        "first_public",
        "last_updated",
        "status",
    ]
    ENA_METADATA_COLUMNS_TO_RENAME = {
        "lat": "latitude_deg",
        "lon": "longitude_deg",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize processor with configuration and directory setup.

        Args:
            config: Configuration dictionary containing processing 
                    parameters such as project directory, primer mode, 
                    and validation settings.
        """
        self.config = config
        self.dirs = dir_utils.SubDirs(self.config["project_dir"])
        self.success: List[Dict[str, Any]] = []
        self.failed: List[Dict[str, str]] = []

    def _determine_target_fragment(self, estimates: Dict[str, str]) -> str:
        """Determine target fragment from primer estimation results.

        Args:
            estimates:          Dictionary mapping target genes 
                                to estimated subfragments.

        Returns:
            target_subfragment: Selected target subfragment 
                                (e.g., 'V4').

        Raises:
            ValueError:         If no valid subfragment can be determined 
                                or if the determined subfragment is not 
                                in DEFAULT_16S_PRIMERS.
        """
        unique = {v for v in estimates.values()}
        if len(unique) == 1:
            target_subfragment = unique.pop()
        elif "16S" in estimates:
            target_subfragment = estimates["16S"]
        else:
            raise ValueError("No valid target subfragment identified")

        if target_subfragment not in constants.DEFAULT_16S_PRIMERS:
            raise ValueError(
                f"Target subfragment '{target_subfragment}' not found in "
                "DEFAULT_16S_PRIMERS. Update configuration or primers "
                "database."
            )
        return target_subfragment

    def _process_group(
        self,
        group: pd.DataFrame,
        dataset: str,
        layout: str,
        platform: str,
        target_subfragment: str,
        fwd_primer: Optional[str],
        rev_primer: Optional[str],
    ) -> Dict[str, Any]:
        """
        Construct parameters dictionary for a metadata group.

        Args:
            group:              Metadata subset for the current group.
            dataset:            Dataset identifier.
            layout:             Library layout (single/paired).
            platform:           Instrument platform.
            target_subfragment: Target 16S subfragment.
            fwd_primer:         Forward primer sequence.
            rev_primer:         Reverse primer sequence.

        Returns:
            Dictionary containing processing parameters for the group.
        """
        return {
            "dataset": dataset,
            "metadata": group,
            "n_runs": len(group),
            "library_layout": layout,
            "instrument_platform": platform,
            "target_subfragment": target_subfragment,
            "pcr_primer_fwd_seq": fwd_primer,
            "pcr_primer_rev_seq": rev_primer,
        }

    def _infer_library_layout(
        self,
        metadata: pd.DataFrame,
        info: Dict[str, Any],  # Unused
    ) -> pd.DataFrame:
        """
        Infer library layout from FASTQ FTP URLs in metadata.

        Args:
            metadata: DataFrame containing sequencing metadata.
            info:     Dataset info dictionary (unused).

        Returns:
            Updated metadata with corrected library_layout column.
        """
        metadata = metadata.copy()
        metadata["fastq_ftp"] = metadata["fastq_ftp"].fillna("")

        url_counts = [
            len([url for url in ftp_urls.strip().split(";") if url])
            for ftp_urls in metadata["fastq_ftp"]
        ]

        library_layout = [
            "paired" if count == 2 else "single" if count == 1 else "unknown"
            for count in url_counts
        ]

        new_layout = pd.Series(
            library_layout, index=metadata.index, name="library_layout"
        )
        original_lower = metadata["library_layout"].str.lower()

        if not original_lower.equals(new_layout.str.lower()):
            mismatches = metadata[original_lower != new_layout.str.lower()]
            if not mismatches.empty:
                mismatches_report = mismatches[['library_layout']].join(
                    new_layout.rename('new_layout')
                )
                logger.debug(
                    f"Library layout mismatch in {len(mismatches)} rows.\n"
                    f"Differences:\n"
                    f"{mismatches_report}"
                )
            metadata["library_layout"] = new_layout
        return metadata

    def _process_citations(self, info: Dict[str, Any]) -> List[str]:
        """
        Extract citations from publication URLs in dataset info.

        Args:
            info: Dataset info dictionary containing publication URLs.

        Returns:
            List of formatted citations or original URLs if citation
            lookup fails.
        """
        citations = []
        urls = str(info.get("publication_url", "")).strip(";").split(";")
        for url in urls:
            if not url:
                continue
            try:
                citation = misc_utils.get_citation(url, style="apa")
                citations.append(citation if citation else url)
            except Exception as e:
                logger.warning(
                    f"Failed to process citation URL {url}: {e}"
                )
                citations.append(url)
        return citations

    def _extract_primers_from_metadata(
        self, meta: pd.DataFrame, info: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract and validate primers from metadata columns.

        Args:
            meta:       Metadata DataFrame.
            info:       Dataset info with potential primer sequences.

        Returns:
            Tuple of validated forward and reverse primer sequences.

        Raises:
            ValueError: If metadata primers conflict with info primers.
        """
        fwd_primer, rev_primer = None, None

        if {"pcr_primer_fwd_seq", "pcr_primer_rev_seq"}.issubset(meta.columns):
            fwd_unique = meta["pcr_primer_fwd_seq"].nunique() == 1
            rev_unique = meta["pcr_primer_rev_seq"].nunique() == 1

            if fwd_unique and rev_unique:
                fwd_primer = meta["pcr_primer_fwd_seq"].iloc[0]
                rev_primer = meta["pcr_primer_rev_seq"].iloc[0]

                info_fwd = info.get("pcr_primer_fwd_seq")
                info_rev = info.get("pcr_primer_rev_seq")

                if info_fwd and info_fwd != fwd_primer:
                    raise ValueError(
                        f"Metadata forward primer {fwd_primer} "
                        f"doesn't match info {info_fwd}"
                    )
                if info_rev and info_rev != rev_primer:
                    raise ValueError(
                        f"Metadata reverse primer {rev_primer} "
                        f"doesn't match info {info_rev}"
                    )

        # Validate primer sequences
        def validate_primer(primer: Optional[str]) -> Optional[str]:
            if not primer:
                return None
            if not re.match(r"^[ACGTURYKMSWBDHVN]+$", primer, re.IGNORECASE):
                raise ValueError(f"Invalid primer sequence: {primer}")
            return primer.upper()

        return (
            validate_primer(fwd_primer or info.get("pcr_primer_fwd_seq")),
            validate_primer(rev_primer or info.get("pcr_primer_rev_seq")),
        )

    def auto(
        self, dataset: str, meta: pd.DataFrame, ena_runs: Dict[str, List[str]]
    ) -> None:
        """
        Automatically estimate primers and process metadata groups.

        Args:
            dataset:    Dataset identifier.
            meta:       Combined metadata DataFrame.
            ena_runs:   

        Raises:
            ValueError: If unable to determine valid primers for the target
                        subfragment.
        """
        estimates = {}
        target_genes = self.config["validate_16s"]["run_targets"]

        for gene in target_genes:
            runs = ena_runs.get(gene, [])
            runs = [run for run in runs if run in meta["run_accession"].values]

            if not runs:
                continue

            results = seq_analyze.estimate_16s_subfragment(
                metadata=meta,
                runs=runs,
                run_label=gene,
                n_runs=self.config["validate_16s"]["n_runs"],
                output_dir=self.dirs.metadata_per_dataset / dataset,
                fastq_dir=self.dirs.seq_data_per_dataset
                / dataset
                / "sequence_validation",
            )

            if gene == "unknown":
                results = {k: v for k, v in results.items() if v[1] >= 10}
            if results:
                ((subfragment, _),) = Counter(results.values()).most_common(1)
                estimates[gene] = subfragment

        try:
            target_subfragment = self._determine_target_fragment(estimates)
        except ValueError as e:
            logger.error(f"Primer estimation failed for {dataset}: {e}")
            raise

        try:
            primers = constants.DEFAULT_16S_PRIMERS[target_subfragment]
            fwd_primer = primers["fwd"]["seq"]
            rev_primer = primers["rev"]["seq"]
        except KeyError:
            raise ValueError(
                f"No primer sequences available for target subfragment "
                f"'{target_subfragment}'"
            )

        group_columns = ["library_layout", "instrument_platform"]
        for (layout, platform), group in meta.groupby(
            group_columns, dropna=False
        ):
            if group.empty:
                continue
            params = self._process_group(
                group,
                dataset,
                layout,
                platform,
                target_subfragment,
                fwd_primer,
                rev_primer,
            )
            self.success.append(params)

    def manual(
        self, dataset: str, info: Dict[str, Any], 
        meta: pd.DataFrame, ena_runs: pd.DataFrame,
    ) -> None:
        """
        Process dataset with manually provided primers and metadata.

        Args:
            dataset:  Dataset identifier.
            info:     Dataset info containing manual configurations.
            meta:     Combined metadata DataFrame.
            ena_runs:
        """
        group_columns = ["library_layout", "instrument_platform"]

        # Primer extraction and validation
        fwd_primer, rev_primer = self._extract_primers_from_metadata(
            meta, info
        )

        # Target subfragment handling
        target_subfragment = info.get("target_subfragment")
        if (
            "target_subfragment" in meta.columns
            and meta["target_subfragment"].nunique() == 1
        ):
            target_subfragment = meta["target_subfragment"].iloc[0]
            group_columns.append("target_subfragment")

        # Handle varying primer sequences in metadata
        if {"pcr_primer_fwd_seq", "pcr_primer_rev_seq"}.issubset(
            meta.columns
        ):
            if (
                meta["pcr_primer_fwd_seq"].nunique() > 1
                or meta["pcr_primer_rev_seq"].nunique() > 1
            ):
                group_columns.extend(
                    ["pcr_primer_fwd_seq", "pcr_primer_rev_seq"]
                )

        # Process each metadata group
        for cols, group in meta.groupby(group_columns, dropna=False):
            if group.empty:
                continue

            # Extract primers from group columns if present
            group_fwd = (
                cols[group_columns.index("pcr_primer_fwd_seq")]
                if "pcr_primer_fwd_seq" in group_columns
                else fwd_primer
            )
            group_rev = (
                cols[group_columns.index("pcr_primer_rev_seq")]
                if "pcr_primer_rev_seq" in group_columns
                else rev_primer
            )

            sample_subset = {
                "dataset": dataset,
                "metadata": group,
                "ena_runs": ena_runs,
                "sample_pooling": info.get("sample_pooling", ""),
                "n_runs": len(group),
                "target_subfragment": (
                    cols[group_columns.index("target_subfragment")]
                    if "target_subfragment" in group_columns
                    else target_subfragment
                ),
                "pcr_primer_fwd_seq": group_fwd,
                "pcr_primer_rev_seq": group_rev,
            }

            # Add remaining grouping columns
            for i, col in enumerate(group_columns):
                if col not in sample_subset:
                    sample_subset[col] = cols[i]

            self.success.append(sample_subset)

    def process(self, dataset: str, info: Dict[str, Any]) -> None:
        """
        Process a dataset with automated or manual primer configuration.

        Args:
            dataset: Dataset identifier (ENA project accession or custom name).
            info:    Dataset configuration parameters.

        Handles:
            - Metadata retrieval and validation
            - Primer determination
            - Error tracking
        """
        try:
            # Attempt to get properly formatted citations for associated publications
            citations = self._process_citations(info)

            # Format and log dataset information
            dataset = dataset.upper()
            dataset_type = info.get('dataset_type', '').upper()
            platform = info.get('instrument_platform', '').upper()
            model = info.get('instrument_model', '')
            layout = info.get('library_layout', '').upper()
            target = f"{info.get('target_gene', '')} {info.get('target_subfragment', '')}".strip()
            fwd_primer = f"{info.get('pcr_primer_fwd', '')} ({info.get('pcr_primer_fwd_seq', '')})"
            rev_primer = f"{info.get('pcr_primer_rev', '')} ({info.get('pcr_primer_rev_seq', '')})"
            publications = citations or ['None']
            
            # Build formatted lines
            WIDTH = 20
            dataset_info = [
                f"\n{'[Dataset]':<{WIDTH}}{dataset}",
                f"{'[Type]':<{WIDTH}}{dataset_type}",
                f"{'[Sequencing Platform]':<{WIDTH}}{platform} ({model})" if platform or model else '',
                f"{'[Library Layout]':<{WIDTH}}{layout}",
                f"{'[Target]':<{WIDTH}}{target}",
                f"{'[Primers]':<{WIDTH}}{fwd_primer}",
                f"{'':<{WIDTH}}{rev_primer}",
                f"{'[Publications]':<{WIDTH}}{publications[0]}"
            ]
            
            # Add additional citations
            dataset_info.extend(f"{'':<{WIDTH}}{cite}" for cite in publications[1:])
            
            # Remove any empty lines and log
            logger.info('\n'.join(line for line in dataset_info if line))

            # ENA metadata retrieval
            # If 'dataset_type' is 'ENA' or the dataset ID matches the ENA pattern
            if (
                self.ENA_PATTERN.match(dataset)
                and info.get("dataset_type", "").upper() == "ENA"
            ):
                ena_data = ENAMetadata(email=self.config["ena_email"])
                ena_data.process_dataset(dataset, info)
                ena_meta = ena_data.df
                ena_runs = ena_data.runs
            else:
                ena_meta = pd.DataFrame()
                ena_runs = {}

            # Manual metadata retrieval
            manual_meta = self.fetch_manual_meta(dataset)

            # Metadata validation and combination
            # If the samples are pooled, restructure the metadata
            if info.get("sample_pooling", ""):
                ena_runs = ena_meta.set_index("run_accession", drop=False)
                parsed_ena_meta = parse_sample_pooling(ena_meta)
                if not manual_meta.empty:
                    similar_rows = calculate_distances(
                        parsed_ena_meta, manual_meta
                    )
                    meta = pd.concat(
                        [parsed_ena_meta.reset_index(drop=True), similar_rows],
                        axis=1,
                    )
                else:
                    meta = parsed_ena_meta
            else:
                meta = self._combine_metadata(
                    dataset, ena_meta, manual_meta, info
                )

            meta = self._infer_library_layout(meta, info)

            # Primer processing mode
            if self.config["pcr_primers_mode"] == "manual":
                run_validate_16s = self.cfg.get("validate_16s", {}).get("enabled", False)
                if run_validate_16s:
                    target_genes = self.config["validate_16s"]["run_targets"]

                    for gene in target_genes:
                        runs = ena_runs.get(gene, [])
                        runs = [run 
                                for run in runs if run in meta["run_accession"].values]
            
                        if not runs:
                            continue
            
                        ena_runs = seq_analyze.validate_16s(
                            metadata=meta,
                            runs=runs,
                            run_label=gene,
                            n_runs=self.config["validate_16s"]["n_runs"],
                            output_dir=self.dirs.metadata_per_dataset 
                            / dataset,
                            fastq_dir=self.dirs.seq_data_per_dataset
                            / dataset
                            / "sequence_validation",
                        )
                        logger.info(ena_runs)
                        
                self.manual(dataset, info, meta, ena_runs)
            else:
                self.auto(dataset, meta, ena_runs)
        except Exception as e:
            logger.error(
                f"Dataset {dataset} failed: {str(e)}", exc_info=True
            )
            self.failed.append({"dataset": dataset, "error": str(e)})

    def _combine_metadata(
        self,
        dataset: str,
        ena_meta: pd.DataFrame,
        manual_meta: pd.DataFrame,
        info: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Combine and validate ENA/manual metadata.

        Args:
            dataset:     Dataset identifier for error reporting.
            ena_meta:    ENA metadata DataFrame.
            manual_meta: Manually provided metadata DataFrame.
            info:        Dataset info for validation.

        Returns:
            combined:    Combined and validated metadata DataFrame.

        Raises:
            ValueError:  For metadata consistency issues.
        """
        if not ena_meta.empty and info.get("dataset_type") != "ENA":
            raise ValueError(
                f"ENA metadata present for non-ENA dataset type: "
                f"{info.get('dataset_type')}"
            )
        if ena_meta.empty and not manual_meta.empty:
            return manual_meta
        elif not ena_meta.empty and manual_meta.empty:
            return ena_meta
        else:
            combined = self.combine_ena_and_manual_metadata(
                dataset, ena_meta, manual_meta
            )
            if combined.empty:
                raise ValueError(
                    f"No valid samples after metadata merge for {dataset}"
                )
            return combined

    def combine_ena_and_manual_metadata(
        self, dataset: str, ena_meta: pd.DataFrame, manual_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge ENA and manual metadata with conflict resolution.

        Args:
            dataset:     Dataset identifier for error reporting.
            ena_meta:    ENA metadata DataFrame.
            manual_meta: Manual metadata DataFrame.

        Returns:
            meta:        Merged metadata DataFrame.

        Raises:
            ValueError:  For critical column mismatches.
        """
        # Standardize column names
        ena_meta.columns = ena_meta.columns.str.lower().str.strip()
        manual_meta.columns = manual_meta.columns.str.lower().str.strip()

        # Validate required columns
        for df, df_type in [(ena_meta, "ENA"), (manual_meta, "manual")]:
            if "run_accession" not in df.columns:
                raise ValueError(
                    f"{df_type} metadata for {dataset} missing "
                    f"'run_accession' column"
                )

        # Resolve column conflicts
        manual_meta, ena_meta = self._resolve_column_conflicts(
            manual_meta, ena_meta
        )

        # Clean ENA metadata
        ena_meta = ena_meta.drop(
            columns=ena_meta.columns.intersection(
                self.ENA_METADATA_UNNECESSARY_COLUMNS
            )
        )
        ena_meta = ena_meta.rename(columns=self.ENA_METADATA_COLUMNS_TO_RENAME)

        # Merge datasets
        meta = manual_meta.merge(
            ena_meta,
            on="run_accession",
            how="left",
            suffixes=("", "_ena"),
            validate="one_to_one",
        )
        if "dataset_id" not in meta.columns:
            meta["dataset_id"] = (
                f"ENA_{dataset}"
                if self.ENA_PATTERN.match(dataset)
                else dataset
            )
        return meta

    def fetch_manual_meta(self, dataset: str) -> pd.DataFrame:
        """
        Retrieve manually-collected metadata for a dataset.

        Args:
            dataset: Dataset identifier.

        Returns:
            DataFrame containing manual metadata. Empty DataFrame 
            if none exists.
        """
        manual_metadata_tsv = (
            Path(self.config["manual_metadata_dir"]) / f"{dataset}.tsv"
        )
        if manual_metadata_tsv.is_file():
            logger.info(
                f"Loading manual metadata from '{manual_metadata_tsv}'..."
            )
            return pd.read_csv(
                manual_metadata_tsv,
                sep="\t",
                encoding="utf8",
                low_memory=False,
                dtype={"run_accession": str},
            )
        return pd.DataFrame()

    @staticmethod
    def _resolve_column_conflicts(
        manual_meta: pd.DataFrame, ena_meta: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resolve column name conflicts between manual and ENA metadata.

        Args:
            manual_meta: Manual metadata DataFrame.
            ena_meta:    ENA metadata DataFrame.

        Returns:
            Tuple of (modified manual_meta, modified ena_meta) with resolved
            conflicts.
        """
        common_cols = set(ena_meta.columns) & set(manual_meta.columns) - {
            "run_accession"
        }
        ena_processed = ena_meta.copy()

        for col in common_cols:
            if manual_meta[col].equals(ena_processed[col]):
                ena_processed = ena_processed.drop(columns=col)
            else:
                ena_processed = ena_processed.rename(
                    columns={col: f"{col}_ena"}
                )
        return manual_meta, ena_processed
