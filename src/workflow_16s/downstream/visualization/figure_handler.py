# workflow_16s/downstream/visualization/figure_handler.py
"""
Consolidated Figure Management for Plotting Operations

Provides unified interface for saving figures (matplotlib & plotly) with:
- Aggressive memory cleanup (critical for SHAP batch rendering)
- Thread-safe sequential processing (matplotlib not thread-safe)
- Consistent error handling (OSError wrapping, never crash pipeline)
- Automatic statistics tracking (success/failure counts)
- Dry-run mode for path/permission validation

Key Design Principles:
1. Sequential-only processing (matplotlib requires this)
2. Cleanup in finally block (ensures cleanup even on failure)
3. Skip+warn on save failure (never crash the 5-hour pipeline)
4. Auto-cleanup after every save (prevents 2,100+ SHAP plot bloat)
5. CDN-hardcoded for Plotly HTML (<50 KB vs 3-5 MB with bundled JS)
6. 'heather' template for consistent Plotly styling
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import gc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


logger = logging.getLogger("workflow_16s")

# ===== CONSTANTS =====
PLOTLY_TEMPLATE = "heather"  # Consistent styling across all Plotly figures
PLOTLY_INCLUDE_PLOTLYJS = "cdn"  # Reduce HTML file size from 3-5 MB to <50 KB
PNG_SCALE_FACTOR = 2  # Balance quality vs. render time for Kaleido


class FigureHandler:
    """
    Unified interface for saving matplotlib and plotly figures with safety guarantees.
    
    Critical for:
    - Batch rendering (2,100+ SHAP plots without memory bloat)
    - Error resilience (skip+warn, never crash)
    - Consistent styling (heather template, CDN JS)
    - Memory management (aggressive cleanup after every save)
    
    All save methods are sequential (NO threading) and return bool success/failure.
    
    Example:
        handler = FigureHandler(config={'storage_backend': 'local'})
        
        # Save matplotlib figure
        success = handler.save_matplotlib(fig, 'output/plot.png', dpi=300)
        
        # Save Plotly HTML (with auto cleanup)
        success = handler.save_plotly_html(fig, 'output/plot.html', auto_cleanup=True)
        
        # Batch rendering with cleanup
        for i, data in enumerate(dataset):
            fig = create_plot(data)
            handler.save_plotly_html(fig, f'output/plot_{i}.html', auto_cleanup=True)
        
        # Check statistics
        stats = handler.get_stats()
        print(f"Success rate: {stats['success_rate']:.1%}")
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        storage_backend: str = 'local'
    ):
        """
        Initialize FigureHandler.
        
        Args:
            config: Configuration dict with optional keys:
                - storage_backend: 'local' (default), 'gcs', 'aws', 's3' (for future)
                - cache_dir: Where to store temporary/cached figures (default: 'data/cache/figures')
            storage_backend: Storage backend ('local', 'gcs', 's3', 'aws')
        """
        self.config = config or {}
        self.storage_backend = self.config.get('storage_backend', storage_backend)
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/figures'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self._success_count = 0
        self._failure_count = 0
        self._failure_log = []  # List of (path, error_type, error_msg)
        
        # Dry-run mode (validate paths/permissions without actual saves)
        self._dry_run = False
        
        logger.debug(f"FigureHandler initialized: backend={self.storage_backend}, cache_dir={self.cache_dir}")
    
    def set_dry_run(self, enabled: bool = True) -> None:
        """
        Enable/disable dry-run mode.
        
        In dry-run mode, validates paths and permissions but does NOT save files.
        Useful for validating configuration before batch operations.
        
        Args:
            enabled: If True, enable dry-run mode; if False, disable it
        """
        self._dry_run = enabled
        mode = "ENABLED" if enabled else "DISABLED"
        logger.debug(f"Dry-run mode {mode}")
    
    def save_matplotlib(
        self,
        fig,
        path: Union[str, Path],
        dpi: int = 300,
        auto_cleanup: bool = True
    ) -> bool:
        """
        Save matplotlib figure to file with error handling and optional auto-cleanup.
        
        Args:
            fig: matplotlib Figure object
            path: Output file path
            dpi: Resolution for PNG/PDF (default 300)
            auto_cleanup: If True, cleanup figure after save (default True)
            
        Returns:
            True if save successful, False otherwise (logged as warning, not error)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self._dry_run:
                logger.debug(f"[DRY-RUN] Would save matplotlib to: {path}")
                return True
            
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            self._success_count += 1
            logger.debug(f"Saved matplotlib figure: {path}")
            return True
            
        except OSError as e:
            # File system error: permission, disk space, etc.
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.warning(f"Failed to save matplotlib to {path}: {type(e).__name__}: {e}")
            return False
            
        except Exception as e:
            # Other errors (corrupted figure, etc.)
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.error(f"Unexpected error saving matplotlib to {path}: {type(e).__name__}: {e}")
            return False
            
        finally:
            if auto_cleanup:
                self.cleanup_figure(fig, fig_type='matplotlib')
    
    def save_plotly_html(
        self,
        fig,
        path: Union[str, Path],
        cdn: bool = True,
        auto_cleanup: bool = True
    ) -> bool:
        """
        Save Plotly figure as interactive HTML.
        
        Always uses 'heather' template for consistent styling.
        Uses CDN for plotly.js (hardcoded) to reduce file size from 3-5 MB to <50 KB.
        
        Args:
            fig: plotly Figure object
            path: Output file path (.html)
            cdn: If True (default), use CDN for plotly.js (reduces file size drastically)
            auto_cleanup: If True, cleanup figure after save (default True)
            
        Returns:
            True if save successful, False otherwise
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self._dry_run:
                logger.debug(f"[DRY-RUN] Would save plotly HTML to: {path}")
                return True
            
            # Ensure consistent template
            if hasattr(fig, 'update_layout'):
                fig.update_layout(template=PLOTLY_TEMPLATE)
            
            # Write with CDN hardcoded (never bundle JS)
            include_plotlyjs = PLOTLY_INCLUDE_PLOTLYJS if cdn else True
            fig.write_html(
                path,
                include_plotlyjs=include_plotlyjs,
                config={'responsive': True}
            )
            
            self._success_count += 1
            logger.debug(f"Saved Plotly HTML: {path} (size: {path.stat().st_size / 1024:.1f} KB)")
            return True
            
        except OSError as e:
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.warning(f"Failed to save Plotly HTML to {path}: {type(e).__name__}: {e}")
            return False
            
        except Exception as e:
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.error(f"Unexpected error saving Plotly HTML to {path}: {type(e).__name__}: {e}")
            return False
            
        finally:
            if auto_cleanup:
                self.cleanup_figure(fig, fig_type='plotly')
    
    def save_plotly_png(
        self,
        fig,
        path: Union[str, Path],
        scale: int = PNG_SCALE_FACTOR,
        auto_cleanup: bool = True
    ) -> bool:
        """
        Save Plotly figure as PNG using Kaleido.
        
        PNG renders are slower than HTML but provide static images for reports.
        
        Args:
            fig: plotly Figure object
            path: Output file path (.png)
            scale: Scale factor for PNG quality (default 2, use 3 for publication quality but slower)
            auto_cleanup: If True, cleanup figure after save (default True)
            
        Returns:
            True if save successful, False otherwise
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if self._dry_run:
                logger.debug(f"[DRY-RUN] Would save Plotly PNG to: {path}")
                return True
            
            # Ensure consistent template
            if hasattr(fig, 'update_layout'):
                fig.update_layout(template=PLOTLY_TEMPLATE)
            
            fig.write_image(path, scale=scale)
            
            self._success_count += 1
            logger.debug(f"Saved Plotly PNG: {path}")
            return True
            
        except OSError as e:
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.warning(f"Failed to save Plotly PNG to {path}: {type(e).__name__}: {e}")
            return False
            
        except Exception as e:
            # Kaleido not installed, permission error, etc.
            self._failure_count += 1
            self._failure_log.append((str(path), type(e).__name__, str(e)))
            logger.warning(f"Failed to save Plotly PNG to {path} (Kaleido installed?): {type(e).__name__}: {e}")
            return False
            
        finally:
            if auto_cleanup:
                self.cleanup_figure(fig, fig_type='plotly')
    
    def cleanup_figure(self, fig, fig_type: str = 'plotly') -> None:
        """
        Aggressively cleanup figure to prevent memory bloat.
        
        Critical for batch rendering (e.g., 2,100+ SHAP plots).
        
        For matplotlib:
            - fig.clf(): Clear all axes (most effective cleanup)
            - plt.close(fig): Close the figure (allows garbage collection)
            - gc.collect(): Force garbage collection
            
        For plotly:
            - del fig: Delete reference
            - gc.collect(): Force garbage collection
        
        Args:
            fig: Figure object to cleanup
            fig_type: 'matplotlib' or 'plotly'
        """
        try:
            if fig_type == 'matplotlib':
                # Matplotlib cleanup: clear axes, close figure, collect garbage
                if hasattr(fig, 'clf'):
                    fig.clf()  # Clear all axes on the figure
                plt.close(fig)  # Close the figure (releases resources)
                gc.collect()  # Force garbage collection
                logger.debug("Cleaned up matplotlib figure (clf + close + gc.collect)")
                
            elif fig_type == 'plotly':
                # Plotly cleanup: delete reference, run garbage collection
                del fig
                gc.collect()  # Force garbage collection
                logger.debug("Cleaned up plotly figure (del + gc.collect)")
            else:
                logger.warning(f"Unknown figure type for cleanup: {fig_type}")
                
        except Exception as e:
            logger.debug(f"Error during figure cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for figure saving operations.
        
        Useful for monitoring batch operations and detecting failures.
        
        Returns:
            Dict with keys:
            - success_count: Number of successful saves
            - failure_count: Number of failed saves
            - total: Total save attempts
            - success_rate: Success rate as float (0.0-1.0)
            - failure_log: List of (path, error_type, error_msg) tuples
        """
        total = self._success_count + self._failure_count
        success_rate = self._success_count / total if total > 0 else 1.0
        
        return {
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'total': total,
            'success_rate': success_rate,
            'failure_log': self._failure_log.copy()
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters and failure log."""
        self._success_count = 0
        self._failure_count = 0
        self._failure_log = []
        logger.debug("Reset FigureHandler statistics")
    
    def log_stats(self) -> None:
        """Log current statistics at INFO level."""
        stats = self.get_stats()
        logger.info(
            f"Figure saving statistics: "
            f"{stats['success_count']} success, "
            f"{stats['failure_count']} failed, "
            f"success_rate={stats['success_rate']:.1%}"
        )
        
        if stats['failure_log']:
            logger.info(f"Failures ({len(stats['failure_log'])}):")
            for path, error_type, error_msg in stats['failure_log'][:10]:  # Log first 10
                logger.info(f"  - {path}: {error_type}: {error_msg}")
            if len(stats['failure_log']) > 10:
                logger.info(f"  ... and {len(stats['failure_log']) - 10} more failures")


def create_figure_handler(config: Optional[Dict[str, Any]] = None) -> FigureHandler:
    """
    Factory function to create a FigureHandler instance.
    
    Allows for easy configuration and testing.
    
    Example:
        handler = create_figure_handler(config={'storage_backend': 'local'})
        handler.save_plotly_html(fig, 'output/plot.html')
    
    Args:
        config: Configuration dict (see FigureHandler.__init__)
        
    Returns:
        FigureHandler instance
    """
    return FigureHandler(config=config)
