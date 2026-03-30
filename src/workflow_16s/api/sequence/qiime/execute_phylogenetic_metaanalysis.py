# file: src/workflow_16s/api/qiime/run_meta_analysis.py

import logging
import subprocess
from pathlib import Path

from workflow_16s.config import AppConfig

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
                # The path is the last element on the line
                return line.split()[-1]
        raise ValueError(f"Conda environment with substring '{env_name_substring}' not found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError("Could not execute 'conda env list'. Is Conda in your PATH?") from e

def execute_phylogenetic_meta_analysis(
    app_config: AppConfig,
    project_dir: Path
) -> Path:
    """
    Constructs and executes the command for the phylogenetic meta-analysis workflow.

    This function calls the `workflow_16s.downstream.meta_analysis` module,
    which harmonizes and merges multiple AnnData objects using QIIME 2's
    fragment insertion.

    Args:
        app_config: The application configuration object, providing paths to references.
        project_dir: The root directory of the project, which should contain the
                     'processed_data' subdirectory with .h5ad files.

    Returns:
        The path to the final, merged AnnData object (.h5ad).
    
    Raises:
        RuntimeError: If the Conda command fails or the subprocess returns an error.
        FileNotFoundError: If the script completes but the expected output file is not found.
    """
    
    # Find the QIIME 2 conda environment path. A generic substring is used for flexibility.
    conda_env_path = get_conda_env_path("qiime2")
    
    # Define the target script module and necessary paths from the config
    script_module = "workflow_16s.downstream.meta_analysis"
    ref_dir = app_config.paths.phylogeny # Assumes config has a 'paths.references' attribute

    # Build the command list for subprocess execution
    command = [
        "conda", "run", "--prefix", conda_env_path,
        "python", "-m", script_module,
        "--project_dir", str(project_dir),
        "--ref_dir", str(ref_dir)
    ]

    try:
        # Format the command for readable logging
        command_str = ' '.join(f'"{c}"' for c in command).replace(" --", " \\\n  --")
        logger.info(f"Executing phylogenetic meta-analysis command:\n{command_str}")
        
        # Run the command, capturing output and checking for errors
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        logger.debug("Meta-analysis STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.warning("Meta-analysis STDERR:\n%s", result.stderr)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Phylogenetic meta-analysis execution failed with code {e.returncode}.")
        logger.error(f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        raise RuntimeError("Phylogenetic meta-analysis workflow failure") from e
        
    # Verify that the expected output file was created and return its path
    output_path = project_dir / "downstream_analysis" / "meta_analysis_merged.h5ad"
    
    if not output_path.exists():
         logger.error(f"Meta-analysis script completed, but expected output was not found at: {output_path}")
         raise FileNotFoundError(f"Expected output file not found: {output_path}")

    logger.info(f"✅ Successfully completed meta-analysis. Final output is available at: {output_path}")
    
    return output_path