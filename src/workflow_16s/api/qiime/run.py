# file: src/workflow_16s/api/qiime/run.py

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

from workflow_16s.config_schema import AppConfig

logger = logging.getLogger("workflow_16s")

def get_conda_env_path(env_name_substring: str) -> str:
    """Finds the full path to a Conda environment by searching for a substring."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith('#') or not line.strip():
                continue
            if env_name_substring in line:
                return line.split()[-1]
        raise ValueError(f"Conda environment with substring '{env_name_substring}' not found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError("Could not execute 'conda env list'. Is Conda in your PATH?") from e

def execute_per_dataset_qiime_workflow(
    app_config: AppConfig,
    subset: Dict[str, Any],
    qiime_dir: Path,
    metadata_path: Path,
    manifest_path: Path
) -> Dict[str, Path]:
    """Constructs and executes the command for the self-contained QIIME 2 workflow."""
    
    # Use a generic name for the conda env for flexibility
    conda_env_path = get_conda_env_path("qiime2")
    q2_config = app_config.qiime2.per_dataset
    script_path = q2_config.script_path
    
    # Determine DADA2 mode from config, defaulting to 'auto' for safety.
    # Assumes config structure like: qiime2.per_dataset.denoise.dada2_mode
    dada2_mode = getattr(q2_config.denoise, 'dada2_mode', 'auto')

    command = [
        "conda", "run", "--prefix", conda_env_path, "python", str(script_path),
        "--dada2-mode", dada2_mode,
        "--qiime_dir", str(qiime_dir),
        "--manifest_tsv", str(manifest_path),
        "--library_layout", str(subset["library_layout"]),
        "--fwd_primer_seq", str(subset["pcr_primer_fwd_seq"]),
    ]

    is_paired = subset["library_layout"] == 'paired'
    if is_paired:
        command.extend(["--rev_primer_seq", str(subset["pcr_primer_rev_seq"])])

    command.extend([
        "--classifier_path", str(app_config.paths.classifier / f"{q2_config.taxonomy.classifier}-classifier.qza"),
        "--chimera_method", q2_config.denoise.chimera_method,
        "--confidence", str(q2_config.taxonomy.confidence), # Corrected from classify_method
        "--n_threads", str(app_config.execution.threads),
        "--min_frequency", str(q2_config.filter.retain_threshold),
        "--collapse_level", str(app_config.collapse_level) # type: ignore
    ])

    # Handle manual DADA2 parameters if mode is 'manual'
    if dada2_mode == 'manual':
        trunc_f = subset.get("trunc_len_f")
        trunc_r = subset.get("trunc_len_r") if is_paired else None
        
        if trunc_f is None or (is_paired and trunc_r is None):
            raise ValueError("Manual DADA2 mode requires 'trunc_len_f' (and 'trunc_len_r' for paired-end) in subset.")

        # Derive trim lengths from primers, mimicking workflow.py logic
        trim_f = len(subset["pcr_primer_fwd_seq"]) if q2_config.trim.enabled else 0
        trim_r = len(subset["pcr_primer_rev_seq"]) if is_paired and q2_config.trim.enabled else 0
        
        dada2_params = [str(trunc_f)]
        if is_paired:
            dada2_params.extend([str(trunc_r), str(trim_f), str(trim_r)])
        else:
            dada2_params.append(str(trim_f))

        command.extend(["--dada2-params", *dada2_params])

    if q2_config.hard_rerun:
        command.append("--hard_rerun")
    if q2_config.trim.enabled:
        command.append("--trim_sequences")

    try:
        command_str = ' '.join(f'"{c}"' for c in command).replace(" --", " \\\n  --")
        logger.info(f"Executing QIIME 2 command:\n{command_str}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug("QIIME STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.warning("QIIME STDERR:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        logger.error(f"QIIME 2 execution failed with code {e.returncode}.")
        logger.error(f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError("QIIME 2 workflow failure") from e

    # Return the paths to the final, key artifacts
    level = app_config.collapse_level # type: ignore
    return {
        "table": qiime_dir / "04_feature-table.qza",
        "rep_seqs": qiime_dir / "04_representative-sequences.qza",
        "taxonomy": qiime_dir / "05_taxonomy.qza",
        "rooted_tree": qiime_dir / "06_rooted-tree.qza",
        "collapsed_table": qiime_dir / f"07_collapsed-table-L{level}.qza",
    }