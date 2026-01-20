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
from plotly.offline import get_plotlyjs_version

from workflow_16s.utils.io import import_js_as_str
from workflow_16s import constants

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
script_dir = Path(__file__).parent  
tables_js_path = script_dir / "tables.js"  
css_path = script_dir / "style.css"  
html_template_path = script_dir / "template.html"  

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


class HTMLReport:
    def __init__(
        self, 
        config: Optional[Dict] = None,
        amplicon_data: "AmpliconData",
    ):  
        self.config = config
        self.amplicon_data = amplicon_data

        self.group_column = config.get('group_column', constants.DEFAULT_GROUP_COLUMN)
        self.group_column_values = config.get("group_column_values", constants.DEFAULT_GROUP_COLUMN_VALUES)
        
        self.figures: Dict = _extract_figures(amplicon_data)
        self.data: Dict = _extract_data(amplicon_data)
      
    def write(
        output_path: Union[str, Path],
        include_sections: Optional[List[str]] = None,
        max_features: int = 20
    ) -> None:
        include_sections = include_sections or [
            k for k, v in self.figures.items() if v
        ]
        if 'violin' in self.figures and 'violin' not in include_sections:
            include_sections.append('violin')

        _prepare_sections()
    
    

    

# ================================== CORE HELPERS =================================== #

def _extract_figures(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    figures = {}
    # SAMPLE MAPS
    if amplicon_data.maps:
        figures['sample_maps'] = amplicon_data.maps
    
    # ALPHA DIVERSITY
    alpha_figures = {}
    for table_type, levels in amplicon_data.alpha_diversity.items():
        for level, data in levels.items():
            if 'figures' in data and data['figures']:
                if table_type not in alpha_figures:
                    alpha_figures[table_type] = {}
                alpha_figures[table_type][level] = data['figures']
    figures['alpha_diversity'] = alpha_figures
        
    # BETA DIVERSITY
    beta_figures = {}
    for table_type, levels in amplicon_data.ordination.items():
        for level, methods in levels.items():
            for method, data in methods.items():
                if data and 'figures' in data and data['figures']:
                    if table_type not in beta_figures:
                        beta_figures[table_type] = {}
                    if level not in beta_figures[table_type]:
                        beta_figures[table_type][level] = {}
                    beta_figures[table_type][level][method] = data['figures']
    figures['beta_diversity'] = beta_figures
    
    # MACHINE LEARNING
    ml_figures = {}
    # 'roc', 'prc', 'confusion_matrix', 'shap_summary_bar', 'shap_summary_beeswarm', 
    # 'shap_summary_heatmap', 'shap_summary_force'
    # 'shap_dependency' (list)
    for table_type, levels in amplicon_data.models.items():
        for level, methods in levels.items():
            for method, model_result in methods.items():
                if model_result and 'figures' in model_result:
                    if table_type not in ml_figures:
                        ml_figures[table_type] = {}
                    if level not in ml_figures[table_type]:
                        ml_figures[table_type][level] = {}
                    ml_figures[table_type][level][method] = model_result['figures']
    figures['ml'] = ml_figures
      
    """
    # Violin plots
    group_1_name = 'contaminated'
    group_2_name = 'pristine'
    violin_figures = {group_1_name: {}, group_2_name: {}}
    for feat in amplicon_data.top_features_group_1:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures[group_1_name][feat['feature']] = feat['violin_figure']
    for feat in amplicon_data.top_features_group_2:
        if 'violin_figure' in feat and feat['violin_figure']:
            violin_figures[group_2_name][feat['feature']] = feat['violin_figure']
    figures['violin'] = violin_figures
    """
    return figures


def _extract_data(amplicon_data: "AmpliconData") -> Dict[str, Any]:
    data = {}

    # MACHINE LEARNING
    ml_data = {}
    # 'model', 'feature_importances', 'top_features', 'best_params', 'test_scores', 'shap_report': shap_report,
    for table_type, levels in amplicon_data.models.items():
        for level, methods in levels.items():
            for method, model_result in methods.items():
                if model_result and 'figures' in model_result:
                    del model_result['figures']
                    if table_type not in ml_data:
                        ml_data[table_type] = {}
                    if level not in ml_data[table_type]:
                        ml_data[table_type][level] = {}
                    ml_data[table_type][level][method] = model_result
    data['ml'] = ml_data
  

def _prepare_sections(
    figures: Dict,
    data: Dict,
    include_sections: List[str],
    id_counter: Iterator[int],
) -> Tuple[List[Dict], Dict]:
  
    sections = []
    plot_data: Dict[str, Any] = {}

    for section in include_sections:
        if section not in figures or section not in data:
            continue
        section_data = {
            'id': f"section-{uuid.uuid4().hex}", 
            'title': section.title(),
            'subsections': []
        }
        if section == "map":
            flat: Dict[str, Any] = {}
            _flatten(figures[section], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(
                    flat, id_counter, section_data['id']
                )
                plot_data.update(pd)
                sec_data["subsections"].append({
                    "title": "Sample Maps",
                    "tabs_html": tabs,
                    "buttons_html": btns
                })

    for sec in include_sections:
        if sec not in figures:
            continue

        sec_data = {
            "id": f"sec-{uuid.uuid4().hex}", 
            "title": sec.title(), 
            "subsections": []
        }

        if sec == "ordination":
            btns, tabs, pd = _ordination_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Ordination",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        
        elif sec == "alpha_diversity":
            btns, tabs, pd = _alpha_diversity_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Alpha Diversity",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        
        elif sec == "map":
            flat: Dict[str, Any] = {}
            _flatten(figures[sec], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(
                    flat, id_counter, sec_data["id"]
                )
                plot_data.update(pd)
                sec_data["subsections"].append({
                    "title": "Sample Maps",
                    "tabs_html": tabs,
                    "buttons_html": btns
                })
        elif sec == "shap":
            btns, tabs, pd = _shap_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "SHAP Interpretability",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        elif sec == 'violin':
            btns, tabs, pd = _violin_to_nested_html(
                figures[sec], id_counter, sec_data["id"]
            )
            plot_data.update(pd)
            sec_data["subsections"].append({
                "title": "Violin Plots",
                "tabs_html": tabs,
                "buttons_html": btns
            })
        else:
            flat: Dict[str, Any] = {}
            _flatten(figures[sec], [], flat)
            if flat:
                tabs, btns, pd = _figs_to_html(
                    flat, id_counter, sec_data["id"], row_label="color_col"
                )
                plot_data.update(pd)
                sec_data["subsections"].append({
                    "title": "All",
                    "tabs_html": tabs,
                    "buttons_html": btns
                })
        
        if sec_data["subsections"]:
            sections.append(sec_data)

    return sections, plot_data

def _flatten(tree: Dict, keys: List[str], out: Dict) -> None:
    for k, v in tree.items():
        new_keys = keys + [k]
        if isinstance(v, dict):
            _flatten(v, new_keys, out)
        else:
            out[" - ".join(new_keys)] = v

def _figs_to_html(
    figs: Dict[str, Any], 
    counter: Iterator[int], 
    prefix: str, 
    *, 
    square: bool = False,
    row_label: Optional[str] = None
) -> Tuple[str, str, Dict]:
    tabs, btns, plot_data = [], [], {}

    for idx, (title, fig) in enumerate(figs.items()):
        pane_id  = f"{prefix}-pane-{next(counter)}"
        plot_id = f"{prefix}-plot-{next(counter)}"

        btns.append(
            f'<button class="tab-button {"active" if idx==0 else ""}" '
            f'data-pane-target="#{pane_id}" '
            f'onclick="showPane(event)">{title}</button>'
        )

        tabs.append(
            f'<div id="{pane_id}" class="tab-pane {"active" if idx==0 else ""}" '
            f'data-plot-id="{plot_id}">'
            f'<div id="container-{plot_id}" class="plot-container"></div></div>'
        )

        try:
            if fig is None:
                raise ValueError("Figure object is None")
                
            if hasattr(fig, "to_plotly_json"):
                pj = fig.to_plotly_json()
                pj.setdefault("layout", {})["showlegend"] = False
                plot_data[plot_id] = {
                    "type": "plotly",
                    "data": pj["data"],
                    "layout": pj["layout"],
                    "square": square
                }
            elif isinstance(fig, Figure):
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                plot_data[plot_id] = {
                    "type": "image",
                    "data": base64.b64encode(buf.read()).decode()
                }
            else:
                plot_data[plot_id] = {
                    "type": "error",
                    "error": f"Unsupported figure type {type(fig)}"
                }
        except Exception as exc:
            logger.exception("Serializing figure failed")
            plot_data[plot_id] = {
                "type": "error", 
                "error": str(exc)
            }
            
    buttons_html = "\n".join(btns)
    if row_label:
        buttons_html = (
            f'<div class="tabs" data-label="{row_label}">'
            f'{buttons_html}</div>'
        )
    else:
        buttons_html = f'<div class="tabs">{buttons_html}</div>'
        
    return "\n".join(tabs), buttons_html, plot_data

def _section_html(sec: Dict) -> str:
    sub_html = "\n".join(
        f'<div class="subsection">\n'
        f'  <h3>{sub["title"]}</h3>\n'
        f'  <div class="tab-content">\n'          
        f'    {sub["buttons_html"]}\n'
        f'    {sub["tabs_html"]}\n'
        f'  </div>\n'                             
        f'</div>'
        for sub in sec["subsections"]
    )
    
    return f'''
    <div class="section" id="{sec["id"]}">
        <div class="section-header" onclick="toggleSection(event)">
            <h2>{sec["title"]}</h2>
            <span class="toggle-icon">▼</span>
        </div>
        <div class="section-content" id="{sec["id"]}-content">
            {sub_html}
        </div>
    </div>
    '''


def _alpha_diversity_to_nested_html(
    figures: Dict[str, Any],
    id_counter: Iterator[int],
    prefix: str,
) -> Tuple[str, str, Dict]:
    buttons_html, panes_html, plot_data = [], [], {}
    
    for t_idx, (table_type, levels) in enumerate(figures.items()):
        table_id = f"{prefix}-table-{next(id_counter)}"
        is_active_table = t_idx == 0
        
        buttons_html.append(
            f'<button class="table-button {"active" if is_active_table else ""}" '
            f'data-pane-target="#{table_id}" '
            f'onclick="showPane(event)">{table_type}</button>'
        )
        
        level_btns, level_panes = [], []
        for l_idx, (level, metrics) in enumerate(levels.items()):
            level_id = f"{table_id}-level-{next(id_counter)}"
            is_active_level = l_idx == 0
            
            level_btns.append(
                f'<button class="level-button {"active" if is_active_level else ""}" '
                f'data-pane-target="#{level_id}" '
                f'onclick="showPane(event)">{level}</button>'
            )
            
            metric_btns, metric_tabs, metric_plot_data = _figs_to_html(
                metrics, id_counter, level_id
            )
            plot_data.update(metric_plot_data)
            
            level_panes.append(
                f'<div id="{level_id}" class="level-pane {"active" if is_active_level else ""}" >'
                f'<div class="tabs" data-label="metric">{metric_btns}</div>'
                f'{metric_tabs}'
                f'</div>'
            )
        
        panes_html.append(
            f'<div id="{table_id}" class="table-pane {"active" if is_active_table else ""}" >'
            f'<div class="tabs" data-label="level">{"".join(level_btns)}</div>'
            f'{"".join(level_panes)}'
            f'</div>'
        )
    
    buttons_row = f'<div class="tabs" data-label="table_type">{"".join(buttons_html)}</div>'
    return buttons_row, "".join(panes_html), plot_data
