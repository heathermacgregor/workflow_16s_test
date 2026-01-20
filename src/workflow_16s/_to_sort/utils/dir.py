# ===================================== IMPORTS ====================================== #

from typing import Any, List, Tuple, Union, Dict
from pathlib import Path
from dataclasses import dataclass
import re
import shutil
import pandas as pd
import logging

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== CLASSES ===================================== #

class Dir:
    """Simple directory management class for basic operations"""
    
    def __init__(self, dir_path: Union[str, Path]):
        self.dir_path = Path(dir_path).resolve()
    
    def create(self, mode: int = 0o755) -> bool:
        """Create directory if it doesn't exist.
        
        Args:
            mode: Directory permissions (default: 755)
            
        Returns:
            True if directory was created or already exists, False on failure
        """
        try:
            if not self.dir_path.exists():
                self.dir_path.mkdir(parents=True, exist_ok=True, mode=mode)
                logger.info(f"Created directory: {self.dir_path}")
                return True
            return True
        except OSError as e:
            logger.error(f"Failed to create directory {self.dir_path}: {e}")
            return False
    
    def remove(self) -> bool:
        """Remove directory and all contents if it exists.
        
        Returns:
            True if directory was removed or doesn't exist, False on failure
        """
        try:
            if self.dir_path.exists():
                if self.dir_path.is_file():
                    self.dir_path.unlink()
                else:
                    shutil.rmtree(self.dir_path)
                logger.info(f"Removed directory: {self.dir_path}")
                return True
            return True
        except OSError as e:
            logger.error(f"Failed to remove directory {self.dir_path}: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if directory exists"""
        return self.dir_path.exists()
    
    def is_empty(self) -> bool:
        """Check if directory is empty"""
        if not self.dir_path.exists():
            return True
        return not any(self.dir_path.iterdir())
    
    def size(self) -> int:
        """Get total size of directory in bytes"""
        if not self.dir_path.exists():
            return 0
        
        total_size = 0
        try:
            for item in self.dir_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size
    
    def list_contents(self, pattern: str = "*") -> List[Path]:
        """List directory contents matching pattern"""
        if not self.dir_path.exists():
            return []
        return list(self.dir_path.glob(pattern))
    
    def __str__(self) -> str:
        return f"Dir({self.dir_path})"
    
    def __repr__(self) -> str:
        return f"Dir(dir_path={self.dir_path!r})"


@dataclass
class DirectoryConfig:
    """Configuration for directory structure"""
    logs: str = 'logs'
    tmp: str = 'tmp'
    data: str = 'data'
    per_dataset: str = 'per_dataset'
    metadata: str = 'metadata'
    seqs: str = 'seqs'
    raw: str = 'raw'
    trimmed: str = 'trimmed'
    qiime: str = 'qiime'
    final: str = 'final_for_adam'
    tables: str = 'tables'
    figures: str = 'figures'


class ProjectDir:
    """Manages project directory structure for bioinformatics pipeline"""
    
    def __init__(self, dir_path: Union[str, Path], config: DirectoryConfig = None, 
                 auto_create: bool = True):
        """Initialize directory structure manager.
        
        Args:
            dir_path: Root directory path
            config: Custom directory configuration
            auto_create: Whether to automatically create directories
        """
        self.main = Path(dir_path).resolve()
        self.config = config or DirectoryConfig()
        
        # Define directory structure
        self._setup_directory_paths()
        
        # Create Dir instances for each path
        self._setup_dir_instances()
        
        if auto_create:
            self.create_dirs()
    
    def _setup_directory_paths(self):
        """Setup all directory paths based on configuration."""
        # First level directories
        self.logs = self.main / self.config.logs
        self.tmp = self.main / self.config.tmp
        self.data = self.main / self.config.data
        self.final = self.main / self.config.final
        
        # Data subdirectories
        self.data_per_dataset = self.data / self.config.per_dataset
        self.metadata_per_dataset = self.data_per_dataset / self.config.metadata
        self.seq_data_per_dataset = self.data_per_dataset / self.config.seqs
        
        # Sequence data subdirectories
        self.raw_seq_data_per_dataset = self.seq_data_per_dataset / self.config.raw
        self.trimmed_seq_data_per_dataset = self.seq_data_per_dataset / self.config.trimmed
        self.qiime_data_per_dataset = self.data_per_dataset / self.config.qiime
        
        # Final output directories
        self.tables = self.final / self.config.tables
        self.figures = self.final / self.config.figures
    
    def _setup_dir_instances(self):
        """Create Dir instances for easier directory management"""
        self.dir_instances = {
            'main': Dir(self.main),
            'logs': Dir(self.logs),
            'tmp': Dir(self.tmp),
            'data': Dir(self.data),
            'data_per_dataset': Dir(self.data_per_dataset),
            'metadata_per_dataset': Dir(self.metadata_per_dataset),
            'seq_data_per_dataset': Dir(self.seq_data_per_dataset),
            'raw_seq_data_per_dataset': Dir(self.raw_seq_data_per_dataset),
            'trimmed_seq_data_per_dataset': Dir(self.trimmed_seq_data_per_dataset),
            'qiime_data_per_dataset': Dir(self.qiime_data_per_dataset),
            'final': Dir(self.final),
            'tables': Dir(self.tables),
            'figures': Dir(self.figures),
        }
    
    @property
    def all_directories(self) -> List[Path]:
        """Return list of all base directories"""
        return [
            self.main,
            self.logs,
            self.tmp,
            self.data,
            self.data_per_dataset,
            self.metadata_per_dataset,
            self.seq_data_per_dataset,
            self.raw_seq_data_per_dataset,
            self.trimmed_seq_data_per_dataset,
            self.qiime_data_per_dataset,
            self.final,
            self.tables,
            self.figures,
        ]
    
    def create_dirs(self, mode: int = 0o755) -> Dict[str, bool]:
        """Create all directories in the structure using Dir instances.
        
        Args:
            mode: Directory permissions (default: 755)
            
        Returns:
            Dictionary mapping directory names to creation success status
        """
        results = {}
        for name, dir_instance in self.dir_instances.items():
            results[name] = dir_instance.create(mode=mode)
        
        successful = sum(results.values())
        logger.info(f"Successfully created/verified {successful}/{len(results)} directories")
        return results
    
    def remove_dirs(self, confirm: bool = False) -> Dict[str, bool]:
        """Remove all directories in the structure.
        
        Args:
            confirm: Safety flag to confirm deletion
            
        Returns:
            Dictionary mapping directory names to removal success status
        """
        if not confirm:
            raise ValueError("Must set confirm=True to remove directories")
        
        results = {}
        # Remove in reverse order (deepest first)
        for name, dir_instance in reversed(list(self.dir_instances.items())):
            if name != 'main':  # Don't remove main directory by default
                results[name] = dir_instance.remove()
        
        return results
    
    def get_dir(self, name: str) -> Dir:
        """Get a Dir instance by name.
        
        Args:
            name: Directory name (e.g., 'logs', 'tmp', 'data')
            
        Returns:
            Dir instance for the specified directory
        """
        if name not in self.dir_instances:
            available = ', '.join(self.dir_instances.keys())
            raise ValueError(f"Unknown directory '{name}'. Available: {available}")
        
        return self.dir_instances[name]
    
    def get_dataset_dirs(self, dataset: str, create: bool = True) -> Dict[str, Dir]:
        """Get dataset-specific directories as Dir instances.
        
        Args:
            dataset: Dataset identifier
            create: Whether to create directories if they don't exist
            
        Returns:
            Dictionary mapping directory names to Dir instances
        """
        if not dataset or not isinstance(dataset, str):
            raise ValueError("Dataset must be a non-empty string")
        
        # Sanitize dataset name for filesystem
        clean_dataset = self._sanitize_name(dataset)
        
        dataset_paths = {
            'tmp': self.tmp / clean_dataset,
            'metadata': self.metadata_per_dataset / clean_dataset,
            'raw_seqs': self.raw_seq_data_per_dataset / clean_dataset,
            'trimmed_seqs': self.trimmed_seq_data_per_dataset / clean_dataset,
            'qiime': self.qiime_data_per_dataset / clean_dataset,
        }
        
        # Create Dir instances
        dataset_dirs = {name: Dir(path) for name, path in dataset_paths.items()}
        
        # Create directories if requested
        if create:
            for dir_instance in dataset_dirs.values():
                dir_instance.create()
        
        return dataset_dirs
    
    def get_subset_dirs(self, subset: Dict[str, str], create: bool = True) -> Dict[str, Dir]:
        """Get subset-specific directories as Dir instances.
        
        Args:
            subset: Dictionary containing subset parameters
                   Required keys: dataset, instrument_platform, library_layout,
                                target_subfragment, pcr_primer_fwd_seq, pcr_primer_rev_seq
            create: Whether to create directories if they don't exist
        
        Returns:
            Dictionary mapping directory names to Dir instances
        """
        required_keys = {
            'dataset', 'instrument_platform', 'library_layout',
            'target_subfragment', 'pcr_primer_fwd_seq', 'pcr_primer_rev_seq'
        }
        
        if not isinstance(subset, dict):
            raise ValueError("Subset must be a dictionary")
        
        missing_keys = required_keys - set(subset.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        
        # Build subset path components
        path_components = [
            self._sanitize_name(subset['dataset']),
            subset['instrument_platform'].lower(),
            subset['library_layout'].lower(),
            subset['target_subfragment'].lower(),
            f"FWD_{self._sanitize_name(subset['pcr_primer_fwd_seq'])}_REV_{self._sanitize_name(subset['pcr_primer_rev_seq'])}"
        ]
        
        base_dirs = {
            'tmp': self.tmp,
            'metadata': self.metadata_per_dataset,
            'raw_seqs': self.raw_seq_data_per_dataset,
            'trimmed_seqs': self.trimmed_seq_data_per_dataset,
            'qiime': self.qiime_data_per_dataset,
        }
        
        subset_paths = {}
        for name, base_dir in base_dirs.items():
            subset_path = base_dir
            for component in path_components:
                subset_path = subset_path / component
            subset_paths[name] = subset_path
        
        # Create Dir instances
        subset_dirs = {name: Dir(path) for name, path in subset_paths.items()}
        
        # Create directories if requested
        if create:
            for dir_instance in subset_dirs.values():
                dir_instance.create()
        
        return subset_dirs
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize names for filesystem compatibility.
        
        Args:
            name: Name to sanitize
            
        Returns:
            Sanitized name safe for filesystem use
        """
        if not name:
            raise ValueError("Name cannot be empty")
        
        # Replace problematic characters with underscores
        sanitized = re.sub(r"[^a-zA-Z0-9\-_.]", "_", name)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_{2,}", "_", sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        
        if not sanitized:
            raise ValueError("Name becomes empty after sanitization")
        
        return sanitized
    
    def exists(self, check_all: bool = False) -> bool:
        """Check if directories exist.
        
        Args:
            check_all: If True, check all directories. If False, only check main directory.
            
        Returns:
            True if directories exist, False otherwise
        """
        if check_all:
            return all(dir_instance.exists() for dir_instance in self.dir_instances.values())
        return self.dir_instances['main'].exists()
    
    def get_disk_usage(self) -> Dict[str, int]:
        """Get disk usage for each directory (in bytes).
        
        Returns:
            Dictionary mapping directory names to sizes in bytes
        """
        return {
            name: dir_instance.size() 
            for name, dir_instance in self.dir_instances.items()
        }
    
    def cleanup_empty_dirs(self) -> int:
        """Remove empty directories from the structure.
        
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        # Sort by depth (deepest first) to avoid removing parent dirs before children
        all_dirs = sorted(self.all_directories, key=lambda x: len(x.parts), reverse=True)
        
        for directory in all_dirs:
            dir_instance = Dir(directory)
            if dir_instance.exists() and directory != self.main:
                try:
                    if dir_instance.is_empty():
                        if dir_instance.remove():
                            removed_count += 1
                except OSError:
                    continue
        
        return removed_count
    
    def get_directory_tree(self, max_depth: int = 3) -> Dict[str, Any]:
        """Get a tree representation of the directory structure.
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            Nested dictionary representing directory tree
        """
        def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth or not path.exists():
                return {}
            
            tree = {}
            try:
                for item in sorted(path.iterdir()):
                    if item.is_dir():
                        tree[item.name] = build_tree(item, current_depth + 1)
                    else:
                        tree[item.name] = f"file ({item.stat().st_size} bytes)"
            except PermissionError:
                tree["<access_denied>"] = {}
                
            return tree
        
        return {self.main.name: build_tree(self.main)}
    
    def __str__(self) -> str:
        """String representation of the directory structure"""
        return f"ProjectDir(main='{self.main}', {len(self.all_directories)} directories)"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"ProjectDir(main={self.main!r}, config={self.config!r})"


# ==================================== FUNCTIONS ===================================== #

def create_project_dirs(project_name: str, base_path: Union[str, Path] = ".", 
                       custom_config: DirectoryConfig = None) -> ProjectDir:
    """Factory function to create a new project directory structure.
    
    Args:
        project_name: Name of the project
        base_path: Base path where project should be created
        custom_config: Custom directory configuration
    
    Returns:
        ProjectDir instance for the project
    """
    project_path = Path(base_path) #/ project_name
    return ProjectDir(project_path, config=custom_config)

'''
# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create project structure
    project = ProjectDir("my_bioinformatics_project")
    
    # Use individual Dir instances
    logs_dir = project.get_dir('logs')
    print(f"Logs directory exists: {logs_dir.exists()}")
    print(f"Logs directory size: {logs_dir.size()} bytes")
    
    # Get dataset directories as Dir instances
    dataset_dirs = project.get_dataset_dirs("experiment_001")
    print("Dataset directories:", {name: str(dir_inst.dir_path) for name, dir_inst in dataset_dirs.items()})
    
    # Check if tmp dataset directory is empty
    tmp_dir = dataset_dirs['tmp']
    print(f"Dataset tmp directory is empty: {tmp_dir.is_empty()}")
    
    # Example subset configuration
    subset_config = {
        'dataset': 'experiment_001',
        'instrument_platform': 'Illumina',
        'library_layout': 'paired',
        'target_subfragment': '16S_V4',
        'pcr_primer_fwd_seq': 'GTGCCAGCMGCCGCGGTAA',
        'pcr_primer_rev_seq': 'GGACTACHVGGGTWTCTAAT'
    }
    
    subset_dirs = project.get_subset_dirs(subset_config)
    print("Subset directories:", {name: str(dir_inst.dir_path) for name, dir_inst in subset_dirs.items()})
    
    # Get directory tree
    tree = project.get_directory_tree()
    print("Directory tree:", tree)


# Example usage and factory functions
def create_project_dirs(project_name: str, base_path: Union[str, Path] = ".", 
                       custom_config: DirectoryConfig = None) -> SubDirs:
    """
    Factory function to create a new project directory structure
    
    Args:
        project_name: Name of the project
        base_path: Base path where project should be created
        custom_config: Custom directory configuration
    
    Returns:
        SubDirs instance for the project
    """
    project_path = Path(base_path) / project_name
    return SubDirs(project_path, config=custom_config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create project structure
    project = SubDirs("my_bioinformatics_project")
    
    # Get dataset directories
    dataset_dirs = project.get_dataset_dirs("experiment_001")
    print("Dataset directories:", dataset_dirs)
    
    # Example subset configuration
    subset_config = {
        'dataset': 'experiment_001',
        'instrument_platform': 'Illumina',
        'library_layout': 'paired',
        'target_subfragment': '16S_V4',
        'pcr_primer_fwd_seq': 'GTGCCAGCMGCCGCGGTAA',
        'pcr_primer_rev_seq': 'GGACTACHVGGGTWTCTAAT'
    }
    
    subset_dirs = project.get_subset_dirs(subset_config)
    print("Subset directories:", subset_dirs)
'''
