# Publication-Ready Plotting Guide (v2.1)

## Overview

Version 2.1 introduces comprehensive publication-quality plotting features to ensure all visualizations meet journal standards out of the box.

## Quick Start

```python
from workflow_16s.downstream.plotting import PlottingUtils, setup_plotting_theme
import logging

# Initialize theme (call once at startup)
setup_plotting_theme()

# Define your analysis parameters
run_settings = {
    'analysis_date': '2026-01-07',
    'pipeline_version': 'v2.1',
    'min_sequencing_depth': 5000,
    'transformation': 'CLR',
    'statistical_test': 'Kruskal-Wallis + FDR',
    'p_threshold': 0.05
}

# Create plotting utility
plot_utils = PlottingUtils(logging.getLogger(__name__), run_settings=run_settings)

# Save any plotly figure
plot_utils.save_plotly_fig(your_figure, output_path)
# Automatically generates: HTML (with settings), PNG (300 DPI), JSON
```

## Features

### 1. Publication Standards

All plots automatically use:
- **Dimensions**: 1200×800 px (journal optimal)
- **Export DPI**: 300 (high-quality print)
- **Typography**: Arial/Helvetica hierarchy
  - Title: 20pt bold
  - Body: 14pt
  - Axis labels: 12pt
- **Margins**: 80px uniform (prevents clipping)
- **Grid**: Light gray horizontal gridlines
- **X-axis**: -45° label rotation (readable)
- **Legend**: Outside plot area (x=1.02)

### 2. Metadata Injection

Every HTML export includes a formatted table with your analysis settings:

```python
run_settings = {
    'analysis_date': '2026-01-07',
    'data_source': 'ENA Public Archive',
    'min_sequencing_depth': 5000,
    'min_prevalence': 2,
    'taxonomic_level': 'Genus',
    'transformation': 'CLR (Centered Log-Ratio)',
    'statistical_test': 'Kruskal-Wallis + Benjamini-Hochberg FDR',
    'p_threshold': 0.05
}

plot_utils = PlottingUtils(logger, run_settings=run_settings)
plot_utils.save_plotly_fig(fig, 'my_analysis')
# my_analysis.html now shows all settings above the plot
```

Benefits:
- Full reproducibility
- Automatic documentation
- No manual record-keeping needed
- Collaborators see exact parameters used

### 3. Custom Legend System

For complex plots or many categories:

```python
from workflow_16s.downstream.plotting import create_custom_legend_annotations

# Replace default legend with custom annotations
categories = ['Control Group', 'Treatment A', 'Treatment B', 'Treatment C']
colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

annotations = create_custom_legend_annotations(
    categories=categories,
    colors=colors,
    title='Experimental Groups',
    x=1.02,      # Position right of plot
    y=1.0,       # Anchor at top
    spacing=0.08 # Vertical spacing between items
)

fig.update_layout(annotations=annotations, showlegend=False)
```

Use cases:
- Plots with >20 categories (default legend auto-hides)
- Custom positioning requirements
- Multi-layered legends (color + shape + size)
- Publication-specific formatting needs

### 4. Smart Legend Management

Legends automatically adjust based on complexity:
- **≤20 categories**: Legend shown outside plot
- **>20 categories**: Legend hidden (use custom annotations)
- **Manual override**: Always available via `create_custom_legend_annotations()`

### 5. Batch Processing

Save multiple plots efficiently:

```python
# Queue plots for parallel saving
for result in analysis_results:
    fig = create_plot(result)
    plot_utils.save_plotly_fig(fig, output_path, batch=True)

# Flush queue (parallel save with max 4 workers)
plot_utils.flush_plot_queue(max_workers=4)
```

## Plot-Specific Settings

### Alpha Diversity

```python
# Scatter plots
fig = px.scatter(df, x='depth', y='diversity', color='group')
fig.update_layout(height=600, width=900)  # Journal standard
fig.update_traces(marker={'size': 10, 'opacity': 0.7})

# Violin plots
fig = px.violin(df, y='diversity', x='group', color='group', box=True, points='outliers')
width = 800 + (n_groups - 1) * 80  # Adaptive width
fig.update_layout(height=600, width=width)
```

### Beta Diversity (PCoA)

```python
fig = px.scatter(
    df, x='PC1', y='PC2', color='group',
    labels={'PC1': f'PC1 ({var[0]:.1f}%)', 'PC2': f'PC2 ({var[1]:.1f}%)'}
)
fig.update_layout(height=700, width=1000)
fig.update_traces(marker={'size': 10, 'opacity': 0.7})
```

### Network Graphs

```python
# Use default 1200×800 for network visualizations
plot_utils.save_plotly_fig(network_fig, output_path)
```

## Output Formats

Every save generates three files:

1. **HTML** (Interactive)
   - Embedded plotly graph
   - Run settings table
   - Download buttons for PNG
   - Full interactivity preserved

2. **PNG** (Publication)
   - 300 DPI (1200×800 → 3600×2400 px)
   - Ready for journal submission
   - No quality loss

3. **JSON** (Data Archive)
   - Complete plotly figure specification
   - Reloadable for modifications
   - Version control friendly

## Best Practices

### DO:
✅ Use `PlottingUtils` for all saves (automatic 300 DPI)
✅ Provide `run_settings` dict (ensures reproducibility)
✅ Let theme handle defaults (pre-optimized for journals)
✅ Use custom legends for >20 categories
✅ Batch queue large numbers of plots

### DON'T:
❌ Save with `fig.write_html()` directly (misses metadata)
❌ Skip `run_settings` (loses reproducibility)
❌ Manually set DPI (theme handles it)
❌ Create legends with >25 items (unreadable)

## Example Workflow

```python
from workflow_16s.downstream.analysis import Analysis
from workflow_16s.downstream.plotting import setup_plotting_theme
import logging

# 1. Setup
setup_plotting_theme()
logger = logging.getLogger(__name__)

# 2. Configure analysis
config = {
    'min_depth': 5000,
    'alpha': 0.05,
    'transformation': 'CLR'
}

# 3. Run analysis
analysis = Analysis(h5ad_path, metadata_path, output_dir, logger, config)
analysis.run_all()

# All plots automatically:
# - Use 300 DPI exports
# - Include run settings in HTML
# - Apply publication theme
# - Generate HTML + PNG + JSON
```

## Migration from v2.0

### Old code:
```python
fig.write_html('output.html')
fig.write_image('output.png')
```

### New code:
```python
from workflow_16s.downstream.plotting import PlottingUtils

plot_utils = PlottingUtils(logger, run_settings=config)
plot_utils.save_plotly_fig(fig, 'output')
# Automatically creates output.html, output.png (300 DPI), output.json
```

## Constants Reference

```python
from workflow_16s.downstream.plotting import (
    DEFAULT_WIDTH,        # 1200 px
    DEFAULT_HEIGHT,       # 800 px
    PUBLICATION_DPI,      # 300
    PLOT_SCALE,           # 3 (for 300 DPI)
    DEFAULT_FONT_SIZE,    # 14 pt
    DEFAULT_TITLE_SIZE,   # 20 pt
    DEFAULT_LABEL_SIZE    # 12 pt
)
```

## Troubleshooting

**Q: My PNG exports are low quality**
A: Ensure you're using `PlottingUtils.save_plotly_fig()`, not `fig.write_image()`. The utility automatically sets `scale=3` for 300 DPI.

**Q: Run settings aren't showing in HTML**
A: Pass `run_settings` dict when creating `PlottingUtils(logger, run_settings={...})`.

**Q: Legend overlaps my plot**
A: For many categories, either use `create_custom_legend_annotations()` or let the auto-hide feature handle it (>20 items).

**Q: Batch saving is slow**
A: Use `batch=True` when calling `save_plotly_fig()`, then call `flush_plot_queue(max_workers=4)` for parallel processing.

## See Also

- [ARCHITECTURE.md](src/workflow_16s/downstream/ARCHITECTURE.md) - Module design
- [/tmp/publication_plotting_example.py](file:///tmp/publication_plotting_example.py) - Complete example
- [plotting.py](src/workflow_16s/downstream/plotting.py) - Source code
