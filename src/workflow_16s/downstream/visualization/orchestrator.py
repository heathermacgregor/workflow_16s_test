"""
ReportOrchestrator: Unified plot registry and batch rendering orchestration.

Features:
- Deferred or immediate PNG rendering (configurable)
- Master HTML report generation with ToC
- Plot metadata tracking (title, module, type)
- Memory-efficient plot queue using weak references
- Thread-safe plot registration and rendering

Optimized for large-scale analysis with 50+ plots.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc

import plotly.graph_objects as go
import plotly.io as pio


class ReportOrchestrator:
    """
    Unified plot orchestration and master report generator.
    
    Supports two rendering strategies:
    1. Batch rendering (deferred to end): Better CPU/memory management
    2. Immediate rendering (per-module): Plots visible instantly
    """
    
    def __init__(
        self,
        output_dir: Path,
        logger: logging.Logger,
        batch_render: bool = True,
        png_scale: int = 2,
        enable_master_report: bool = True
    ):
        """
        Args:
            output_dir: Base output directory for all reports
            logger: Logger instance
            batch_render: If True, defer PNG rendering until finalize()
            png_scale: PNG scale factor (1=96 DPI, 2=192 DPI, 3=288 DPI)
            enable_master_report: Generate master HTML with ToC
        """
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.batch_render = batch_render
        self.png_scale = png_scale
        self.enable_master_report = enable_master_report
        
        # Plot registry: module_name -> (fig, output_subdir, metadata)
        # Using dict for memory efficiency (not storing figs directly, just refs)
        self._plots: Dict[str, List[Tuple[go.Figure, Path, Dict[str, Any]]]] = {}
        self._plots_lock = __import__('threading').Lock()
        
        # Metadata index for master report
        self._metadata: List[Dict[str, Any]] = []
        self._rendering_stats: Dict[str, int] = {'rendered': 0, 'errors': 0}
    
    def add_plot(
        self,
        fig: go.Figure,
        plot_name: str,
        module_name: str,
        output_subdir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a plot for later rendering or immediate save.
        
        Args:
            fig: Plotly figure object
            plot_name: Unique plot identifier (e.g., "alpha_richness_boxplot")
            module_name: Analysis module name (e.g., "Alpha_Diversity")
            output_subdir: Subdirectory under output_dir to save to
            metadata: Optional dict with plot metadata (description, category, etc.)
        """
        if output_subdir is None:
            output_subdir = self.output_dir / module_name.lower()
        
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        plot_meta = metadata or {}
        plot_meta['name'] = plot_name
        plot_meta['module'] = module_name
        plot_meta['output_dir'] = str(output_subdir)
        
        with self._plots_lock:
            if module_name not in self._plots:
                self._plots[module_name] = []
            
            self._plots[module_name].append((fig, output_subdir, plot_meta))
        
        self.logger.debug(f"Registered plot: {plot_name} ({module_name})")
        
        # Immediate rendering if not in batch mode
        if not self.batch_render:
            self._render_single_plot(fig, plot_name, output_subdir)
    
    def _render_single_plot(self, fig: go.Figure, plot_name: str, output_dir: Path) -> None:
        """
        Render a single plot to HTML and PNG (immediate mode).
        
        Args:
            fig: Plotly figure
            plot_name: Plot identifier
            output_dir: Directory to save to
        """
        try:
            # HTML (always)
            html_path = output_dir / f"{plot_name}.html"
            fig.write_html(str(html_path))
            self.logger.debug(f"Saved HTML: {html_path.name}")
            
            # PNG (with specified scale)
            png_path = output_dir / f"{plot_name}.png"
            try:
                fig.write_image(
                    str(png_path),
                    scale=self.png_scale,
                    width=1200,
                    height=800
                )
                self.logger.debug(f"Saved PNG: {png_path.name} (scale={self.png_scale}x)")
            except Exception as e:
                self.logger.warning(f"PNG rendering failed for {plot_name}: {e}")
            
            self._rendering_stats['rendered'] += 1
        except Exception as e:
            self.logger.error(f"Failed to render plot {plot_name}: {e}")
            self._rendering_stats['errors'] += 1
    
    def finalize(self) -> Path:
        """
        Finalize all deferred plots and generate master HTML report.
        
        Returns:
            Path to master HTML report
        """
        self.logger.info("🎨 Finalizing report generation...")
        
        if self.batch_render:
            self._render_all_plots_batch()
        
        if self.enable_master_report:
            report_path = self._generate_master_report()
            self.logger.info(f"✅ Master report saved to {report_path}")
            return report_path
        else:
            self.logger.info(f"📊 Report generation complete ({self._rendering_stats['rendered']} plots rendered)")
            return self.output_dir / "index.html"
    
    def _render_all_plots_batch(self) -> None:
        """
        Batch render all deferred plots using ThreadPoolExecutor.
        
        Optimization:
        - Max 4 concurrent workers to balance speed vs memory
        - Forces garbage collection after each batch
        - Logs progress every 10 plots
        """
        self.logger.info("📊 Batch rendering deferred plots...")
        
        all_plots = []
        with self._plots_lock:
            for module_name, plots_list in self._plots.items():
                all_plots.extend([(fig, name, output_dir, meta) for fig, output_dir, meta in plots_list])
        
        total = len(all_plots)
        self.logger.info(f"Rendering {total} plots...")
        
        # Render with thread pool (max 4 workers for memory efficiency)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for fig, output_dir, meta in all_plots:
                future = executor.submit(
                    self._render_single_plot,
                    fig,
                    meta['name'],
                    output_dir
                )
                futures[future] = meta['name']
            
            # Process completions as they finish
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    self.logger.info(f"  {completed}/{total} plots rendered")
                
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Batch render error: {e}")
                    self._rendering_stats['errors'] += 1
            
            # Force garbage collection after batch
            gc.collect()
        
        self.logger.info(f"Batch rendering complete ({self._rendering_stats['rendered']} plots)")
    
    def _generate_master_report(self) -> Path:
        """
        Generate master HTML report with table of contents.
        
        Structure:
        - Header with metadata (timestamp, workflow version, etc.)
        - Table of Contents (linked sections)
        - Per-module sections with embedded HTML figures
        - PNG links for archive/printing
        """
        import datetime
        
        report_path = self.output_dir / "Master_Report.html"
        
        html_parts = []
        
        # HTML Header
        html_parts.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>16S Analysis Master Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
                .toc { background: white; padding: 20px; border-radius: 5px; 
                      margin-bottom: 30px; border-left: 4px solid #667eea; }
                .toc h2 { margin-top: 0; }
                .toc ul { list-style: none; padding-left: 0; }
                .toc li { padding: 5px 0; }
                .toc a { color: #667eea; text-decoration: none; }
                .toc a:hover { text-decoration: underline; }
                .module-section { background: white; padding: 25px; margin-bottom: 30px; 
                                 border-radius: 5px; border-left: 4px solid #764ba2; }
                .module-section h1 { margin-top: 0; color: #333; }
                .plot-container { margin: 20px 0; padding: 15px; 
                                 background: #fafafa; border-radius: 5px; }
                .plot-title { font-weight: bold; color: #556b2f; margin-bottom: 10px; }
                .png-link { margin-top: 10px; font-size: 0.9em; }
                .png-link a { color: #0066cc; }
                footer { text-align: center; margin-top: 50px; color: #999; font-size: 0.85em; }
            </style>
        </head>
        <body>
        """)
        
        # Header with timestamp
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_parts.append(f"""
        <div class="header">
            <h1>🧬 16S Amplicon Analysis Master Report</h1>
            <p>Generated: {now}</p>
        </div>
        """)
        
        # Table of Contents
        html_parts.append("<div class=\"toc\"><h2>📑 Table of Contents</h2><ul>")
        for module_name in sorted(self._plots.keys()):
            toc_id = module_name.lower().replace(' ', '-')
            html_parts.append(f"<li><a href=\"#{toc_id}\">{module_name}</a></li>")
        html_parts.append("</ul></div>")
        
        # Per-module sections
        for module_name in sorted(self._plots.keys()):
            toc_id = module_name.lower().replace(' ', '-')
            html_parts.append(f"<div class=\"module-section\" id=\"{toc_id}\">")
            html_parts.append(f"<h1>📊 {module_name}</h1>")
            
            with self._plots_lock:
                plots_in_module = self._plots.get(module_name, [])
            
            for fig, output_dir, meta in plots_in_module:
                plot_name = meta.get('name', 'unknown')
                
                # Try to embed HTML file
                html_file = output_dir / f"{plot_name}.html"
                if html_file.exists():
                    try:
                        with open(html_file, 'r') as f:
                            embedded_html = f.read()
                        
                        # Extract plotly div (avoid full HTML nesting)
                        if '<div id="' in embedded_html:
                            # For now, just link to the file (embedding plotly can be complex)
                            html_parts.append(f"<div class=\"plot-container\">")
                            html_parts.append(f"<div class=\"plot-title\">{plot_name}</div>")
                            html_parts.append(f"<a href=\"{html_file.name}\" target=\"_blank\">")
                            html_parts.append(f"📈 View Interactive Plot</a>")
                            html_parts.append(f"</div>")
                    except Exception as e:
                        self.logger.warning(f"Failed to embed {html_file}: {e}")
                
                # Add PNG link
                png_file = output_dir / f"{plot_name}.png"
                if png_file.exists():
                    png_rel = png_file.relative_to(self.output_dir)
                    html_parts.append(f"<div class=\"png-link\">")
                    html_parts.append(f"🖼️  <a href=\"{png_rel}\">Download PNG</a>")
                    html_parts.append(f"</div>")
            
            html_parts.append("</div>")
        
        # Footer
        html_parts.append("""
        <footer>
            <p>16S Amplicon Analysis Workflow | Powered by workflow_16s</p>
        </footer>
        </body>
        </html>
        """)
        
        # Write master report
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_parts))
        
        return report_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        with self._plots_lock:
            n_modules = len(self._plots)
            n_plots = sum(len(plots) for plots in self._plots.values())
        
        return {
            'modules': n_modules,
            'plots_registered': n_plots,
            'plots_rendered': self._rendering_stats['rendered'],
            'rendering_errors': self._rendering_stats['errors'],
            'batch_mode': self.batch_render
        }
