# ===================================== IMPORTS ====================================== #

import base64
import itertools
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pprint import pformat
from plotly.offline import get_plotlyjs_version
import plotly.io as pio

from workflow_16s.utils.io import import_js_as_str

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
logger.setLevel(logging.DEBUG)  # Set to DEBUG for better troubleshooting

script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ===================================== CLASSES ====================================== #

class NumpySafeJSONEncoder(json.JSONEncoder):
    def default(self, obj) -> Any:  
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
        
# ========================== PLOTLY SELECTOR INTEGRATION ========================== #

def _generate_plotly_selector_html(
    figures_dict: Dict[str, Any], 
    container_id: str = "plotly-container",
    section_title: str = "Plots"
) -> str:
    """Generate HTML with interactive selection UI for a nested dictionary of Plotly figures."""
    
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = ' > ') -> Dict[str, Any]:
        """Flatten nested dictionary and create display labels"""
        if d is None:  # Handle None input
            return {}
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if v is None:  # Skip None values
                continue
            if isinstance(v, dict) and not hasattr(v, 'to_plotly_json'):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def create_selector_options(d: Dict, parent_key: str = '', level: int = 0) -> str:
        """Create hierarchical option elements for the selector"""
        options_html = ""
        indent = "  " * level
        
        for key, value in d.items():
            full_key = f"{parent_key} > {key}" if parent_key else key
            
            if isinstance(value, dict) and not hasattr(value, 'to_plotly_json'):
                # Create optgroup for nested dictionaries
                options_html += f'{indent}<optgroup label="{key}">\n'
                options_html += create_selector_options(value, full_key, level + 1)
                options_html += f'{indent}</optgroup>\n'
            else:
                # Create option for figure
                options_html += f'{indent}<option value="{full_key}">{key}</option>\n'
        
        return options_html
    
    # Flatten the dictionary to get all figures with their paths
    flat_figures = flatten_dict(figures_dict)
    
    if not flat_figures:
        return f'<div id="{container_id}">No valid figures found in {section_title.lower()}.</div>'
    
    # Get the first figure to display initially
    first_key = list(flat_figures.keys())[0]
    
    # Convert all figures to JSON using existing conversion function
    figures_json = {}
    for key, fig in flat_figures.items():
        serialized = _convert_figure_to_serializable(fig)
        # Use custom encoder for all figures
        figures_json[key] = json.dumps(serialized, cls=NumpySafeJSONEncoder)  # Add encoder here
    
    # Create the selector options HTML
    selector_options = create_selector_options(figures_dict)
    
    # Generate the HTML with integrated styling
    html_template = f"""
    <div id='{container_id}' class='plotly-selector-container'>
        <div class='selector-controls'>
            <label for='{container_id}-selector' class='selector-label'>Select {section_title}:</label>
            <select id='{container_id}-selector' class='figure-dropdown'>{selector_options}</select>
        </div>
        <div id='{container_id}-plot' class="plotly-selector-plot"></div>
    </div>
        
    <script>
    (function() {{
        // Store figures data for {container_id}
        const figuresData_{container_id.replace('-', '_')} = {json.dumps(figures_json)};
            
        // Get DOM elements
        const selector = document.getElementById('{container_id}-selector');
        const plotDiv = document.getElementById('{container_id}-plot');
            
        // Function to display a plot
        function displayPlot_{container_id.replace('-', '_')}(figureKey) {{
            if (figuresData_{container_id.replace('-', '_')}[figureKey]) {{
                const figureData = JSON.parse(figuresData_{container_id.replace('-', '_')}[figureKey]);
                    
                if (figureData.type === 'plotly') {{
                    // Handle Plotly figures
                    Plotly.newPlot(plotDiv, figureData.data, figureData.layout, {{responsive: true}});
                }} else if (figureData.type === 'image') {{
                    // Handle matplotlib/image figures
                    plotDiv.innerHTML = `<img src="data:image/png;base64,${{figureData.data}}" style="max-width: 100%; height: auto;" alt="Plot">`;
                }} else if (figureData.type === 'error') {{
                    // Handle errors
                    plotDiv.innerHTML = `<div class="error-message">Error loading figure: ${{figureData.error}}</div>`;
                }}
            }}
        }}
            
        // Event listener for selector change
        selector.addEventListener('change', function() {{
            displayPlot_{container_id.replace('-', '_')}(this.value);
        }});
            
        // Display initial plot
        if (selector.value) {{
            displayPlot_{container_id.replace('-', '_')}(selector.value);
        }}
    }})();
    </script>
    """
    return html_template

# ================================== CORE HELPERS =================================== #

shap_fig_titles = {
    "roc": "ROC",
    "prc": "PRC",
    "confusion_matrix": "Confusion Matrix",
    "shap_summary_bar": "SHAP Summary (Bar)",
    "shap_summary_beeswarm": "SHAP Summary (Beeswarm)",
    "shap_summary_heatmap": "SHAP Summary (Heatmap)",
    "shap_summary_force": "SHAP Summary (Force)"
}

from collections import defaultdict
from typing import Dict, Any, List, Union


def _extract_figures(amplicon_data: Any) -> Dict[str, Any]:
    """Extract figures from amplicon data across different analysis types.
    
    Args:
        amplicon_data: AmpliconData object containing analysis results
        
    Returns:
        Dictionary containing organized figures by analysis type
    """
    figures = {
        'ordination': _extract_ordination_figures(amplicon_data),
        'alpha_diversity': _extract_alpha_diversity_figures(amplicon_data),
        'shap': _extract_shap_figures(amplicon_data),
        'violin': _extract_violin_figures(amplicon_data)
    }
    
    # Extract sample maps (simple case)
    if getattr(amplicon_data, 'maps', None):
        figures['map'] = amplicon_data.maps
        logger.info(f"Sample maps extracted: {len(amplicon_data.maps)} maps")
    
    return figures


def _create_nested_defaultdict(depth: int):
    """Create nested defaultdict of specified depth."""
    if depth == 1:
        return defaultdict(dict)
    return defaultdict(lambda: _create_nested_defaultdict(depth - 1))


def _extract_ordination_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    """Extract ordination figures from amplicon data using defaultdict for efficiency."""
    ordination_data = getattr(amplicon_data, 'ordination', None)
    if not ordination_data:
        return {}
    
    # 5-level nested structure: group_column -> table_type -> level -> method -> color_col
    ordination_figures = _create_nested_defaultdict(4)
    
    for group_column, table_types in ordination_data.items():
        if not isinstance(table_types, dict):
            continue
        for table_type, levels in table_types.items():
            if not isinstance(levels, dict):
                continue
            for level, level_data in levels.items():
                figures = _get_nested_value(level_data, ['figures'])
                if not figures:
                    continue
                for method, method_figures in figures.items():
                    if isinstance(method_figures, dict):
                        ordination_figures[table_type][level][method].update(method_figures)
                        color_cols = list(method_figures.keys())
                        logger.info(f"Ordination figures extracted: {group_column}/{table_type}/{level}/{method} "
                                  f"({len(color_cols)} color columns)")
    
    return _defaultdict_to_dict(ordination_figures)


def _extract_alpha_diversity_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    """Extract alpha diversity figures using defaultdict for efficiency."""
    alpha_data = getattr(amplicon_data, 'alpha_diversity', None)
    if not alpha_data:
        return {}
    
    # group_column -> table_type -> level
    alpha_figures = _create_nested_defaultdict(2)
    
    for group_column, table_types in alpha_data.items():
        if not isinstance(table_types, dict):
            continue
        for table_type, levels in table_types.items():
            if not isinstance(levels, dict):
                continue
            for level, data in levels.items():
                figures = _get_nested_value(data, ['figures'])
                if figures:
                    alpha_figures[group_column][table_type][level] = figures
                    logger.info(f"Alpha diversity figures extracted: {group_column}/{table_type}/{level}")
    
    return _defaultdict_to_dict(alpha_figures)


def _extract_shap_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    """Extract SHAP figures using defaultdict for efficiency."""
    models_data = getattr(amplicon_data, 'models', None)
    if not models_data:
        return {}
    
    # group_column -> table_type -> level -> method
    shap_figures = _create_nested_defaultdict(3)
    
    for group_column, table_types in models_data.items():
        if not isinstance(table_types, dict):
            continue
        for table_type, levels in table_types.items():
            if not isinstance(levels, dict):
                continue
            for level, methods in levels.items():
                if not isinstance(methods, dict):
                    continue
                for method, result in methods.items():
                    figures = _get_nested_value(result, ['figures'])
                    if figures:
                        transformed_figures = _transform_shap_figures_efficient(figures)
                        shap_figures[group_column][table_type][level][method] = transformed_figures
                        logger.info(f"SHAP figures extracted: {group_column}/{table_type}/{level}/{method}")
    
    return _defaultdict_to_dict(shap_figures)


def _transform_shap_figures_efficient(figures: Dict[str, Any]) -> Dict[str, Any]:
    """Efficiently transform SHAP figures using dictionary comprehensions."""
    transformed = {}
    for key, val in figures.items():
        if key == 'shap_dependency':
            if isinstance(val, list):
                transformed.update({f'SHAP (Dependency - {i})': fig for i, fig in enumerate(val)})
            elif isinstance(val, dict):
                transformed.update({f'SHAP (Dependency - {feature})': fig for feature, fig in val.items()})
            else:
                transformed[key] = val
        elif key in shap_fig_titles.keys():
            new_key = shap_fig_titles[key]
            transformed[new_key] = val
        else:
            transformed[key] = val
    return transformed


def _extract_violin_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    """Extract violin figures using defaultdict for efficiency."""
    top_features = getattr(amplicon_data, 'top_features', None)
    if not top_features:
        return {}
    violin_figures = defaultdict(dict)
    for col, vals in top_features.items():
        if not isinstance(vals, dict):
            continue
        feature_count = 0
        for val, features in vals.items():
            if not isinstance(features, (list, tuple)):
                continue
            for feature in features:
                if (isinstance(feature, dict) and 
                    'violin_figure' in feature and 
                    'feature' in feature):
                    
                    violin_figures[col][feature['feature']] = feature['violin_figure']
                    feature_count += 1
        
        if feature_count > 0:
            logger.info(f"Violin figures extracted: {col} ({feature_count} features)")
    
    return dict(violin_figures)


def _get_nested_value(data: Any, keys: List[str], default=None) -> Any:
    """Efficiently get nested value from dictionary-like object.
    
    Args:
        data: Dictionary or object to search
        keys: List of keys to traverse
        default: Default value if not found
        
    Returns:
        Nested value or default
    """
    if not isinstance(data, dict):
        return default
    
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current if current else default


def _defaultdict_to_dict(dd: Union[defaultdict, dict]) -> dict:
    """Recursively convert defaultdict to regular dict for JSON serialization.
    
    Args:
        dd: defaultdict or dict to convert
        
    Returns:
        Regular dictionary
    """
    if isinstance(dd, defaultdict):
        dd = dict(dd)
    
    for key, value in dd.items():
        if isinstance(value, defaultdict):
            dd[key] = _defaultdict_to_dict(value)
        elif isinstance(value, dict):
            dd[key] = _defaultdict_to_dict(value)
    
    return dd


def _convert_figure_to_serializable(fig, min_height=1000):
    """Convert figure object to serializable dict with minimum height enforcement."""
    try:
        if fig is None:
            return {"type": "error", "error": "Figure object is None"}
            
        if hasattr(fig, "to_plotly_json"):
            pj = fig.to_plotly_json()
            layout = pj.setdefault("layout", {})
            layout["showlegend"] = False
            
            # Enforce minimum height
            current_height = layout.get("height")
            if current_height is None or current_height < min_height:
                layout["height"] = min_height
            
            return {
                "type": "plotly",
                "data": pj["data"],
                "layout": layout,
                "square": False
            }
        elif isinstance(fig, Figure):
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            buf.seek(0)
            return {
                "type": "image",
                "data": base64.b64encode(buf.read()).decode()
            }
        else:
            return {
                "type": "error",
                "error": f"Unsupported figure type {type(fig)}"
            }
    except Exception as exc:
        logger.exception("Serializing figure failed")
        return {"type": "error", "error": str(exc)}


def _flatten_figures_tree(
    tree: Dict, 
    prefix: str = "", 
    delimiter: str = " - "
) -> List:
    """Flatten a nested figures tree into a list of (path, figure) tuples."""
    flat = []
    if not isinstance(tree, dict):
        return [("", tree)]
    
    for key, value in tree.items():
        new_prefix = f"{prefix}{delimiter}{key}" if prefix else key
        if isinstance(value, dict):
            flat.extend(_flatten_figures_tree(value, new_prefix, delimiter))
        else:
            flat.append((new_prefix, value))
    return flat


def _prepare_sections(
    figures: Dict,
    include_sections: List[str],
) -> List[Dict]:
    sections = []

    for section in include_sections:
        if section not in figures or not figures[section]:
            continue

        # Use the new Plotly selector for this section
        container_id = f"plotly-selector-{section}"
        section_html = _generate_plotly_selector_html(
            figures[section], 
            container_id, 
            section.title()
        )

        section_data = {
            "id": f"sec-{uuid.uuid4().hex}", 
            "title": section.title(), 
            "html_content": section_html  # Store the complete HTML
        }

        sections.append(section_data)

    return sections


def _section_html(section: Dict) -> str:
    """Generate section HTML using the integrated Plotly selector"""
    id = section["id"]
    title = section["title"]
    content = section["html_content"]
    html = f'''
    <div class="section" id="{id}">
        <div class="section-header" onclick="toggleSection(event)"><h2>{title}</h2><span class="toggle-icon">▼</span></div>
        <div class="section-content" id="{id}-content">
            <div class="subsection">{content}</div>
        </div>
    </div>
    '''
    return html
    

def _prepare_features_table(
    features: List[Dict], 
    max_features: int,
    category: str
) -> pd.DataFrame:
    if not features: # Handle empty features 
        return pd.DataFrame({"Feature": [f"No significant {category} features found"]})
    
    if not isinstance(max_features, int): # Validate max_features type
        raise TypeError("max_features must be an integer")

    # Create DataFrame and limit to max_features
    df = pd.DataFrame(features[:max_features])
    
    # Column renaming mapping
    rename_map = {
        "feature": "Feature",
        "level": "Taxonomic Level",
        "test": "Test",
        "effect": "Effect Size",
        "p_value": "P-value",
        "effect_dir": "Direction"
    }
    
    # Rename existing columns only
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_renames)
    
    # Process faprotax_functions if present
    if "faprotax_functions" in df.columns:
        df["Functions"] = df["faprotax_functions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
        df = df.drop(columns=["faprotax_functions"])
    
    # Format numeric columns safely
    numeric_columns = ["Effect Size", "P-value"]
    for col in numeric_columns:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == "Effect Size":
                df[col] = df[col].fillna(0).apply(lambda x: f"{x:.4f}")
            elif col == "P-value":
                df[col] = df[col].fillna(1).apply(lambda x: f"{x:.2e}")
    
    # Define and select output columns
    output_columns = ["Feature", "Taxonomic Level", "Test", "Effect Size", 
                      "P-value", "Direction", "Functions"]
    available_columns = [col for col in output_columns if col in df.columns]
    
    return df[available_columns]


def _prepare_stats_summary(stats: Dict) -> pd.DataFrame:
    summary = []
    for column, tables in stats.items():
        for table_type, levels in tables.items():
            for level, tests in levels.items():
                for test_name, df in tests.items():
                    if isinstance(df, pd.DataFrame) and "p_value" in df.columns:
                        n_sig = sum(df["p_value"] < 0.05)
                    else:
                        n_sig = 0
                    summary.append({
                        "Column": column,
                        "Table Type": table_type,
                        "Test": test_name,
                        "Level": level,
                        "Significant Features": n_sig,
                        "Total Features": len(df) if isinstance(df, pd.DataFrame) else 0
                    })
    
    return pd.DataFrame(summary)


def _validate_models_structure(models: dict):
    """Log structure of models dictionary for debugging"""
    logger.debug("Models structure:\n" + pformat(
        {k1: {k2: list(v2.keys()) for k2, v2 in v1.items()} 
         for k1, v1 in models.items()}, depth=3))


def _prepare_ml_summary(
    models: Dict, 
    top_group_1: List[Dict], 
    top_group_2: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if not models:
        return pd.DataFrame(), pd.DataFrame(), {}

    metrics_summary, features_summary, shap_summary = [], [], {}
    for group_column, table_types in models.items():
        if not isinstance(table_types, dict): # Check that table_types is a dict
            continue
        for table_type, levels in table_types.items():
            if not isinstance(levels, dict): # Check that levels is a dict
                continue
            for level, methods in levels.items():
                if not isinstance(methods, dict): # Check that methods is a dict
                    continue
                for method, result in methods.items():
                    logger.info(f"Reading {group_column}/{table_type}/{level}/{method} results")
                    if not result or not isinstance(result, dict):
                        logger.warning(f"Invalid result for {group_column}/{table_type}/{level}/{method}")
                        continue
                        
                    if "test_scores" not in result:
                        logger.error(f"Missing 'test_scores' in {table_type}/{level}/{method}")
                        continue
                        
                    if "top_features" not in result:
                        logger.error(f"Missing 'top_features' in {table_type}/{level}/{method}")
                        continue
                    
                    test_scores = result["test_scores"]
                    metrics = {
                        "Column": group_column,
                        "Table Type": table_type,
                        "Level": level,
                        "Method": method,
                        "Top Features (N)": len(result.get("top_features", [])),
                        "Accuracy": f"{test_scores.get('accuracy', 'N/A')}",
                        "F1 Score": f"{test_scores.get('f1', 'N/A')}",
                        "MCC": f"{test_scores.get('mcc', 'N/A')}",
                        "ROC AUC": f"{test_scores.get('roc_auc', 'N/A')}",
                        "PR AUC": f"{test_scores.get('pr_auc', 'N/A')}"
                    }
                    metrics_summary.append(metrics)
                    
                    feat_imp = result.get("feature_importances", {})
                    top_features = result.get("top_features", [])[:20]
                    for i, feat in enumerate(top_features, 1):
                        importance = feat_imp.get(feat, 0)
                        features_summary.append({
                            "Feature": feat,
                            "Importance": f"{importance:.4f}" if isinstance(importance, (int, float)) else "N/A",
                            "Column": group_column,
                            "Table Type": table_type,
                            "Level": level,
                            "Method": method,
                            "Rank": i                            
                        })
                    
                    if "shap_report" in result:
                        key = (table_type, level, method)
                        shap_summary[key] = result["shap_report"]
    
    metrics_df = pd.DataFrame(metrics_summary) if metrics_summary else pd.DataFrame()
    features_df = pd.DataFrame(features_summary) if features_summary else pd.DataFrame()
    return metrics_df, features_df, shap_summary


def _prepare_shap_table(shap_summary: Dict) -> pd.DataFrame:
    """Prepare comprehensive SHAP data table for ML section using DataFrame."""
    rows = []
    for (table_type, level, method), df in shap_summary.items():
        # Check if df is a DataFrame and has data
        if isinstance(df, pd.DataFrame) and not df.empty: 
            df_copy = df.copy()
            # Add model identifier columns
            df_copy["Table Type"] = table_type
            df_copy["Level"] = level
            df_copy["Method"] = method
            rows.append(df_copy)
    
    if not rows:
        return pd.DataFrame(columns=[
            "Table Type", "Level", "Method", "Feature", "Mean |SHAP|", 
            "Spearman's ρ", "Beeswarm Interpretation", 
            "Dependency Strength", "Dependency Trend", "Partner Feature", 
            "Interaction Strength", "Relationship"
        ])
    
    # Combine all DataFrames
    combined_df = pd.concat(rows, ignore_index=True)
    
    # Rename columns for better display
    column_mapping = {
        "feature": "Feature",
        "mean_abs_shap": "Mean |SHAP|",
        "beeswarm_correlation": "Spearman's ρ",
        "beeswarm_direction": "Beeswarm Interpretation",
        "dependency_strength": "Dependency Strength",
        "dependency_trend": "Dependency Trend",
        "interaction_partner": "Partner Feature",
        "interaction_strength": "Interaction Strength",
        "relationship_type": "Relationship"
    }
    combined_df = combined_df.rename(columns=column_mapping)
    return combined_df
    

def _format_ml_section(
    ml_metrics: pd.DataFrame, 
    ml_features: pd.DataFrame,
    shap_reports: Dict  
) -> str:
    if ml_metrics is None or ml_metrics.empty:
        return "<p>No ML results available</p>"
    try:
        ml_metrics_html = ml_metrics.to_html(index=False, classes='dynamic-table', table_id='ml-metrics-table')
        
        tooltip_map = {
            "MCC": "Balanced classifier metric (-1 to 1) that considers all confusion matrix values...",
            "ROC AUC": "Probability that random positive ranks higher than random negative...",
            "F1 Score": "Balance between precision and recall...",
            "PR AUC": "Positive-class focused metric for imbalanced data..."
        }
        ml_metrics_html = _add_header_tooltips(ml_metrics_html, tooltip_map)
        
        enhanced_metrics = f"""
        <div class="table-container" id="container-ml-metrics-table">
            {ml_metrics_html}
            <div class="table-controls">
                <div class="pagination-controls">
                    <span>Rows per page:</span>
                    <select class="rows-per-page" onchange="changePageSize('ml-metrics-table', this.value)">
                        <option value="5">5</option>
                        <option value="10" selected>10</option>
                        <option value="20">20</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="-1">All</option>
                    </select>
                    <div class="pagination-buttons" id="pagination-ml-metrics-table"></div>
                    <span class="pagination-indicator" id="indicator-ml-metrics-table"></span>
                </div>
            </div>
        </div>
        """
        
        features_html = _add_table_functionality(ml_features, 'ml-features-table') if ml_features is not None and not ml_features.empty else "<p>No feature importance data available</p>"
        
        # SHAP Analysis table
        shap_html = ""
        if shap_reports:
            shap_df = _prepare_shap_table(shap_reports)
            if not shap_df.empty:
                shap_html = """
                <h3>SHAP Analysis</h3>
                <p>Comprehensive SHAP analysis for top features across all models:</p>
                """ + _add_table_functionality(shap_df, 'shap-table')

        ml_section_html = f"""
        <div class="ml-section">
            <h3>Model Performance</h3>
            {enhanced_metrics}
                
            <h3>Top Features by Importance</h3>
            {features_html}
                
            {shap_html}
        </div>
        """
        return ml_section_html
        
    except Exception as e:
        logger.exception("Error formatting ML section")
        return f"<div class='error'>Error in ML section: {str(e)}</div>"
        

def _add_header_tooltips(
    table_html: str, 
    tooltip_map: Dict[str, str]
) -> str:
    for header, tooltip_text in tooltip_map.items():
        tooltip_html = (
            f'<span class="tooltip">{header}'
            f'<span class="tooltiptext">{tooltip_text}</span>'
            f'</span>'
        )
        table_html = table_html.replace(f'<th>{header}</th>', f'<th>{tooltip_html}</th>')
    return table_html


def _sanitize_table_id(table_id: str) -> str:
    """Replace problematic characters in table IDs for CSS compatibility."""
    return table_id.replace('=', '_eq_').replace(' ', '_')


def _add_table_functionality(df: pd.DataFrame, table_id: str) -> str:
    if df is None or df.empty:
        return "<p>No data available</p>"
    
    # Sanitize table ID
    sanitized_id = _sanitize_table_id(table_id)
    container_id = f"container-{sanitized_id}"
    pagination_id = f"pagination-{sanitized_id}"
    indicator_id = f"indicator-{sanitized_id}"
    
    table_html = df.to_html(index=False, classes='dynamic-table', table_id=sanitized_id)

    table = f"""
    <div class='table-container' id='{container_id}'>
        {table_html}
        <div class='table-controls'>
            <div class='pagination-controls'>
                <span>Rows per page:</span>
                <select class='rows-per-page' onchange="changePageSize('{sanitized_id}', this.value)">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="20">20</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                    <option value="-1">All</option>
                </select>
                <div class='pagination-buttons' id='{pagination_id}'></div>
                <span class='pagination-indicator' id='{indicator_id}'></span>
            </div>
        </div>
    </div>
    """
    return table


def _prepare_advanced_stats_section(advanced_results: Dict) -> str:
    """Prepare HTML section for advanced statistical analysis results."""
    if not advanced_results:
        return "<p>No advanced statistical analysis results available.</p>"
    
    html_content = "<div class='advanced-stats-section'>"
    #html_content += "<h3>Advanced Statistical Analyses</h3>"
    
    # Core Microbiome Results
    if 'core_microbiome' in advanced_results:
        html_content += "<h4>Core Microbiome Analysis</h4>"
        core_data = []
        core_dfs = []
        for group_col, table_types in advanced_results['core_microbiome'].items():
            for table_type, levels in table_types.items():
                for level, groups in levels.items():
                    for group_value, df in groups.items():
                        if isinstance(df, pd.DataFrame):
                            core_data.append({
                                "Group Column": group_col,
                                "Table Type": table_type,
                                "Level": level,
                                "Group Value": group_value,
                                "Core Features": len(df)
                            })
                            df['group_column'] = group_col
                            df['table_type'] = table_type
                            df['level'] = level
                            core_dfs.append(df)
        if core_data:
            core_df = pd.DataFrame(core_data)
            html_content += _add_table_functionality(core_df, 'core-microbiome-table')
        if core_dfs:
            core_df2 = pd.concat(core_dfs, axis=0)
            html_content += _add_table_functionality(core_df2, 'core-microbiome2-table')
        else:
            html_content += "<p>No core microbiome results</p>"
    
    # Correlation Results
    if 'correlations' in advanced_results:
        html_content += "<h4>Correlation Analysis</h4>"
        corr_data = []
        sig_data = []
        for var, table_types in advanced_results['correlations'].items():
            for table_type, levels in table_types.items():
                for level, df in levels.items():
                    if isinstance(df, pd.DataFrame):
                        num_sig = (df['p_value'] < 0.05).sum() if 'p_value' in df.columns else 0
                        corr_data.append({
                            "Column": var,
                            "Table Type": table_type,
                            "Level": level,
                            "Tested": len(df),
                            "Significant (p<0.05)": num_sig
                        })
                        sig_df = df[df['p_value'] < 0.05][['feature', 'rho', 'p_value', 'n_samples']]
                        sig_df['group_column'] = var
                        sig_df['table_type'] = table_type
                        sig_df['level'] = level
                        sig_data.append(sig_df)
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            html_content += _add_table_functionality(corr_df, 'correlation-table')
        if sig_data:
            sig_df_final = pd.concat(sig_data, axis=0)
            # Manipulate the df so we are getting significant features across tests
            sig_df_feature_counts = sig_df_final.value_counts("feature", ascending=False).to_frame()
            sig_df_feature_counts["feature"] = sig_df_feature_counts.index
            html_content += _add_table_functionality(sig_df_feature_counts, 'correlation-features-table')
        else:
            html_content += "<p>No correlation results</p>"
    
    # Network Analysis Results
    if 'networks' in advanced_results:
        html_content += "<h4>Microbial Network Analysis</h4>"
        network_data = []
        network_dfs = []
        for table_type, levels in advanced_results['networks'].items():
            for level, methods in levels.items():
                for method, data in methods.items():
                    if 'edges' in data and isinstance(data['edges'], pd.DataFrame):
                        edges = data['edges']
                        network_data.append({
                            "Method": method,
                            "Table Type": table_type,
                            "Level": level,
                            "Edges": len(edges),
                            "Positive Edges": (edges['correlation'] > 0).sum(),
                            "Negative Edges": (edges['correlation'] < 0).sum()
                        })
                        edges["Method"] = method
                        edges["Table Type"] = table_type
                        edges["Level"] = level
                        network_dfs.append(edges)
                        
        if network_data:
            network_df = pd.DataFrame(network_data)
            html_content += _add_table_functionality(network_df, 'network-table')
        if network_dfs:
            network_df = pd.concat(network_dfs, axis=0)
            network_df = network_df[network_df['abs_correlation'] > 0.8].sort_values(by=['abs_correlation', 'edge_type'])
            html_content += _add_table_functionality(network_df, 'network-table2')
        else:
            html_content += "<p>No network analysis results</p>"
    
    # Summary Report
    if 'summary_path' in advanced_results:
        html_content += f"""
        <h4>Comprehensive Analysis Report</h4>
        <p>Detailed report available at: <code>{advanced_results['summary_path']}</code></p>
        """
    
    html_content += "</div>"
    return html_content


def generate_html_report(
    amplicon_data: "AmpliconData",
    output_path: Union[str, Path],
    include_sections: Optional[List[str]] = None,
    max_features: int = 20,
    config: Optional[Dict] = None
) -> None:
    if config:
        group_col = config.get("group_column", "nuclear_contamination_status")
        group_col_values = config.get("group_column_values", [True, False])
    else:
        group_col = "nuclear_contamination_status"
        group_col_values = [True, False]
        
    figures_dict = _extract_figures(amplicon_data)
    include_sections = include_sections or [
        k for k, v in figures_dict.items() if v
    ]
    if 'violin' in figures_dict and 'violin' not in include_sections:
        include_sections.append('violin')
    
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start tables HTML with overall summary section
    tables_html = "<div class=\"subsection\">"
    
    # Add overall statistical summary section
    if amplicon_data.stats and isinstance(amplicon_data.stats, dict) and 'summary' in amplicon_data.stats:
        summary = amplicon_data.stats['summary']
        
        # Create overall stats table
        overall_data = [
            {"Metric": "Total tests run", "Value": summary['total_tests_run']},
            {"Metric": "Group columns analyzed", "Value": ", ".join(summary['group_columns_analyzed'])}
        ]
        df_overall = pd.DataFrame(overall_data)
        overall_table1_html = _add_table_functionality(df_overall, 'overall-stats-table1')
        
        # Create test-specific stats table
        test_data = []
        for test, count in summary['significant_features_by_test'].items():
            effect_stats = summary['effect_sizes_summary'].get(test, {})
            test_data.append({
                "Test": test,
                "Significant ": count,
                "Effect Size Mean": effect_stats.get('mean', 'N/A'),
                "Effect Size Std.": effect_stats.get('std', 'N/A'),
                "Effect Size Min.": effect_stats.get('min', 'N/A'),
                "Effect Size Max.": effect_stats.get('max', 'N/A')
            })
        df_by_test = pd.DataFrame(test_data)
        overall_table2_html = _add_table_functionality(df_by_test, 'overall-stats-table2')
        
        # Top features across tests
        if 'top_features' in amplicon_data.stats and not amplicon_data.stats['top_features'].empty:
            top_features_df = amplicon_data.stats['top_features']
            overall_top_features_html = _add_table_functionality(top_features_df, 'overall-top-features-table')
        else:
            overall_top_features_html = "<p>No overall top features data</p>"
        
        # Recommendations
        if amplicon_data.stats and isinstance(amplicon_data.stats, dict) and 'recommendations' in amplicon_data.stats:
            recs = amplicon_data.stats['recommendations']
            rec_html = "<ul>"
            for rec in recs:
                rec_html += f"<li>{rec}</li>"
            rec_html += "</ul>"
        else:
            rec_html = "<p>No recommendations</p>"

        # Add advanced statistical analysis section if available
        if amplicon_data.stats and isinstance(amplicon_data.stats, dict) and 'comprehensive_analysis' in amplicon_data.stats:
            advanced_html = _prepare_advanced_stats_section(amplicon_data.stats['comprehensive_analysis'])
            tables_html += """
            <div class="subsection">
                <h3>Advanced Statistical Analyses</h3>
                {advanced_html}
            </div>
            """.format(advanced_html=advanced_html)
            
        tables_html += f"""
        <h3>Overall Analysis Summary</h3>
        
        <h4>Summary Statistics</h4>
        {overall_table1_html}
        
        <h4>Test-Specific Summary</h4>
        {overall_table2_html}
        
        <h4>Top Features Across All Tests</h4>
        {overall_top_features_html}
        
        <h4>Analysis Recommendations</h4>
        {rec_html}
        """
    else:
        tables_html += "<p>No overall statistical summary available.</p>"
    
    tables_html += "</div>"  # Close overall summary subsection
    
    # Start group-specific analysis section
    tables_html += "<div class=\"subsection\">"
    tables_html += "<h3>Group-Specific Analysis</h3>"
    
    # Loop through top_features for group-specific features
    for source, source_dict in amplicon_data.top_features.items():
        for col, val_dict in source_dict.items():
            for val, features in val_dict.items():
                group_key = f"{col}={val}"
                try:
                    df = _prepare_features_table(features, max_features, group_key)
                    logger.info(type(df))
                    logger.info(df)
                    tables_html += f"""
                    <h4>Features associated with {group_key}</h4>
                    {_add_table_functionality(df, f'{group_key}-table')}
                    """
                except Exception as e:
                    logger.error(f"Failed to get features table: {e}")
    
    # Stats summary (per-test details)
    if amplicon_data.stats and isinstance(amplicon_data.stats, dict) and 'test_results' in amplicon_data.stats:
        try:
            stats_df = _prepare_stats_summary(amplicon_data.stats['test_results'])
        except Exception as e:
            logger.error(f"Failed to get stats summary: {e}")
    else:
        stats_df = pd.DataFrame()
    
    tables_html += f"""
    <h3>Statistical Test Summary</h3>
    {_add_table_functionality(stats_df, 'stats-table')}
    """
    
    # ML summary
    if amplicon_data.models:
        _validate_models_structure(amplicon_data.models)
        ml_metrics, ml_features, shap_reports = _prepare_ml_summary(amplicon_data.models, [], [])
        logger.info("Got ML data")
    else:
        ml_metrics, ml_features, shap_reports = pd.DataFrame(), pd.DataFrame(), {}
    
    tables_html += f"""
    <h3>Machine Learning Results</h3>
    {_format_ml_section(ml_metrics, ml_features, shap_reports)}
    """
    
    tables_html += "</div>"  # Close group-specific subsection
    
    sections = _prepare_sections(figures_dict, include_sections)
    sections_html = "\n".join(_section_html(s) for s in sections)

    nav_items = [
        ("Analysis Summary", "analysis-summary"),
        *[(sec['title'], sec['id']) for sec in sections]
    ]
    
    nav_html = """
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
    """
    for title, section_id in nav_items:
        nav_html += f'<li><a href="#{section_id}">{title}</a></li>\n'
    nav_html += "        </ul>\n    </div>"

    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    plotly_js_tag = (
        f'<script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>'
    )
    
    try:
        table_js = import_js_as_str(tables_js_path)
    except Exception as e:
        logger.error(f"Error reading JavaScript file: {e}")
        table_js = ""
    
    try:
        css_content = css_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading CSS file: {e}")
        css_content = ""
        
    try:
        html_template = html_template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error loading HTML template: {e}")
        html_template = """<!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>Report generation failed: Missing template</body>
        </html>"""

    html = html_template.format(
        title="16S Amplicon Analysis Report",
        plotly_js_tag=plotly_js_tag,
        generated_ts=ts,
        section_list=", ".join(include_sections),
        nav_html=nav_html,
        tables_html=tables_html,
        sections_html=sections_html,
        plot_data_json="{}",  # Empty since selectors handle their own data
        table_js=table_js,
        css_content=css_content
    )
        
    output_path.write_text(html, encoding="utf-8")

# ========================== ADDITIONAL UTILITY FUNCTIONS ========================== #

def create_standalone_plotly_selector(
    figures_dict: Dict[str, Any], 
    title: str = "Interactive Plotly Dashboard",
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Create a standalone HTML page with just the Plotly selector."""
    
    selector_html = _generate_plotly_selector_html(figures_dict, "main-dashboard", title)
    
    try:
        plotly_ver = get_plotlyjs_version()
    except Exception:
        plotly_ver = "3.0.1"
    
    complete_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-{plotly_ver}.min.js"></script>
        <style>
            body {{
                margin: 20px;
                background-color: #ffffff;
                font-family: Arial, sans-serif;
            }}
            
            .plotly-selector-container {{
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            .selector-controls {{
                margin-bottom: 15px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .selector-label {{
                font-weight: bold;
                margin: 0;
            }}
            
            .figure-dropdown {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
                min-width: 250px;
                font-size: 14px;
            }}
            
            .plotly-selector-plot {{
                width: 100%;
                min-height: 1000px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            
            .error-message {{
                padding: 20px;
                color: #721c24;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                border-radius: 4px;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <h1 style="text-align: center; color: #333;">{title}</h1>
        {selector_html}
    </body>
    </html>
    """
    
    if output_path:
        Path(output_path).write_text(complete_html, encoding="utf-8")
        logger.info(f"Standalone Plotly selector saved to {output_path}")
    
    return complete_html
