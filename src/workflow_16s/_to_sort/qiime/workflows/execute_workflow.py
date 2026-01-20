# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.io import missing_files

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PER_DATASET = (
    Path(os.path.abspath(__file__)).parent.parent 
    / "src" / "workflow_16s" / "qiime" / "workflows" / "per_dataset_run.py"
)

# ==================================== FUNCTIONS ===================================== #

def get_conda_env_path(env_name_substring: str) -> str:
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        for line in result.stdout.splitlines():
            if env_name_substring in line:
                return line.split()[-1]
        raise ValueError(
            f"Conda environment containing '{env_name_substring}' not found."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error finding conda environment: {e}")
      

def execute_per_dataset_qiime_workflow(
    cfg: Dict[str, Any],
    subset: Dict[str, Union[str, Path, bool, dict]],
    qiime_dir: Union[str, Path],
    metadata_path: Union[str, Path],
    manifest_path: Union[str, Path],
    default_script_path: Union[str, Path] = DEFAULT_PER_DATASET,
) -> Dict[str, Path]:
    qiime_env_path = get_conda_env_path("qiime2-amplicon-2024.10")
    qiime_config = cfg["qiime2"]["per_dataset"]
    
    # Get configured script path and check existence
    script_path = Path(qiime_config["script_path"])
    if not script_path.exists():
        logger.warning(f"Script not found at '{script_path}', using default")
        script_path = default_script_path
    
    command = [
        "conda", "run",
        "--prefix", qiime_env_path,
        "python", str(script_path),  
        "--qiime_dir", str(qiime_dir),
        "--metadata_tsv", str(metadata_path),
        "--manifest_tsv", str(manifest_path),
        "--library_layout", str(subset["library_layout"]).lower(),
        "--instrument_platform", str(subset["instrument_platform"]).lower(),
        "--fwd_primer", str(subset["pcr_primer_fwd_seq"]),
        "--rev_primer", str(subset["pcr_primer_rev_seq"]),
        "--classifier_dir", str(qiime_config["taxonomy"]["classifier_dir"]),
        "--classifier", str(qiime_config["taxonomy"]["classifier"]),
        "--classify_method", 
        str(qiime_config["taxonomy"]["classify_method"]).lower(),
        "--retain_threshold", str(qiime_config["filter"]["retain_threshold"]),
        "--chimera_method", str(qiime_config["denoise"]["chimera_method"]),
        "--denoise_algorithm", str(qiime_config["denoise"]["denoise_algorithm"]),
    ]
    
    if qiime_config.get("hard_rerun", False):
        command.append("--hard_rerun")
    if qiime_config.get("trim", {}).get("enabled", False):
        command.append("--trim_sequences")

    try:
        command_str = ' '.join(command).replace(" --", " \\\n--")
        command_str = command_str.replace(" python ", " \\\npython ")
        logger.info(f"\nExecuting QIIME2 command:\n{command_str}")
        result = subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        logger.debug("QIIME STDOUT:\n%s", result.stdout)
        logger.debug("QIIME STDERR:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"QIIME2 execution failed with code {e.returncode}:\n"
            f"Command: {e.cmd}\nError output:\n{e.stderr}"
        )
        logger.error(error_msg)
        raise RuntimeError("QIIME2 workflow failure") from e

    expected_outputs = [
        metadata_path,
        manifest_path,
        qiime_dir / "table" / "feature-table.biom",
        qiime_dir / "rep-seqs" / "dna-sequences.fasta",
        qiime_dir / qiime_config["taxonomy"]["classifier"] / "taxonomy" / "taxonomy.tsv",
        qiime_dir / "table_6" / "feature-table.biom",
    ]
    missing_outputs = missing_files(expected_outputs)
    if missing_outputs:
        missing_outputs_txt = '\n'.join(['  • ' + str(item) for item in missing_outputs])
        raise RuntimeError(f"Missing required QIIME outputs: \n{missing_outputs_txt}")
        
    return {
        "metadata": metadata_path,
        "manifest": manifest_path,
        "table": expected_outputs[2],
        "rep_seqs": expected_outputs[3],
        "taxonomy": expected_outputs[4],
        "table_6": expected_outputs[5],
    }
  
