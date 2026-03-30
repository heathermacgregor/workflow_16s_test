# workflow_16s/dashboard/server.py

import json
import logging
import os
from pathlib import Path
from flask import Flask, jsonify, render_template, Response
import yaml

from .monitor import SystemMonitor
from .data_viewer import DataViewer

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 🔧 FIXED: Read from config.yaml or environment variable
CONFIG_PATH = Path(os.getenv(
    "WORKFLOW_16S_CONFIG",
    "/usr2/people/macgregor/amplicon/workflow_16s/config/config.yaml"
))

def get_project_root() -> Path:
    """
    Determine project root from (in priority order):
    1. WORKFLOW_16S_PROJECT_DIR environment variable
    2. config.yaml paths.project setting
    3. Fallback to hardcoded default
    """
    # Priority 1: Environment variable
    env_project = os.getenv("WORKFLOW_16S_PROJECT_DIR")
    if env_project:
        return Path(env_project)
    
    # Priority 2: Config file
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data and 'paths' in config_data and 'project' in config_data['paths']:
                    return Path(config_data['paths']['project'])
    except Exception as e:
        logging.warning(f"Could not load config from {CONFIG_PATH}: {e}")
    
    # Priority 3: Fallback default
    return Path("/usr2/people/macgregor/amplicon/project_01")

PROJECT_ROOT = get_project_root()  # 🔧 FIXED: Dynamic resolution

app = Flask(__name__)
monitor = SystemMonitor(PROJECT_ROOT)
viewer = DataViewer(PROJECT_ROOT)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/config')
def config_editor():
    return render_template("config.html")

@app.route('/api/telemetry')
def telemetry():
    state = {}
    state_file = PROJECT_ROOT / "workflow_state.json"
    if state_file.exists():
        try:
            with open(state_file, 'r') as f: state = json.load(f)
        except: pass

    return jsonify({
        "processes": monitor.get_process_tree(),
        "recent_files": monitor.get_recent_files(),
        "cache_counts": monitor.get_cache_counts(),
        "workflow_state": state,
        "h5ad_stats": viewer.get_stats(),
        "hot_sites": viewer.get_nfc_hot_sites()
    })

@app.route('/api/data_viz')
def data_viz():
    return jsonify({
        "map_points": viewer.get_map_data(),
        "composition": viewer.get_composition()
    })

@app.route('/api/stream_logs')
def stream_logs():
    return Response(monitor.yield_log_lines(), mimetype='text/event-stream')

@app.route('/api/get_config')
def get_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return jsonify({"yaml": f.read()})
    return jsonify({"yaml": ""})

@app.route('/api/provenance')
def get_provenance():
    return jsonify(viewer.get_provenance_stats())

def start_server(host='0.0.0.0', port=5005):
    print(f"🚀 AmpliScout Dashboard running at http://{host}:{port}")
    # 🟢 FIXED: Explicitly enable threaded mode so the log stream doesn't block API calls
    app.run(host=host, port=port, debug=True, use_reloader=False, threaded=True)