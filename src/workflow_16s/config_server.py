import os
import sys
import time
import json
import psutil
import yaml
import threading
import re
import smtplib
import math
import traceback
from email.message import EmailMessage
import pandas as pd
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Response

app = Flask(__name__)

# 🟢 PATHS & CONFIG
PROJECT_ROOT = Path("/auto/sahara/namib/home/macgregor/amplicon/project_01")
LOG_DIR = PROJECT_ROOT / "07_logs"
STATE_FILE = PROJECT_ROOT / "workflow_state.json"
FACILITIES_FILE = PROJECT_ROOT / "01_raw_data" / "_nfc_facilities" / "facilities.tsv"
CONFIG_FILE = Path("/auto/sahara/namib/home/macgregor/amplicon/workflow_16s/config/config.yaml")
SMS_CONFIG_FILE = PROJECT_ROOT / "sms_config.json"

SERVER_START_TIME = time.time()
SYS_HISTORY = []  
PROC_CACHE = {} 

H5AD_STATS = {
    "total_files": 0, "total_samples": 0, "points": [], "metadata_composition": {}, "available_columns": [], "status": "Initializing..." 
}

FACILITY_POINTS = []
LAST_SMS_TIME = 0
SMS_STATE = {}

# ==========================================
# 🟢 CORE UTILITY FUNCTIONS
# ==========================================

def clean_dict(obj):
    """Recursively purges NaN/Infinity values and protects against Numpy Array crashes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return 0.0
        return obj
    elif isinstance(obj, dict):
        return {str(k): clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_dict(v) for v in obj]
    
    # Safely check for Pandas NaNs without triggering Truth Value ambiguity on arrays
    if type(obj) in (int, float, str, bool, type(None)):
        try:
            if pd.isna(obj): return None
        except: pass
    return obj

def safe_float(val):
    """Safely converts coordinates, catching empty strings, N/A, and NaNs."""
    try:
        if pd.isna(val) or str(val).strip() == '' or str(val).strip().lower() in ['nan', 'n/a', 'none']:
            return None
        return float(val)
    except Exception:
        return None

def load_facilities():
    global FACILITY_POINTS
    if FACILITIES_FILE.exists():
        try:
            df = pd.read_csv(FACILITIES_FILE, sep='\t')
            FACILITY_POINTS = []
            for _, row in df.iterrows():
                if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                    FACILITY_POINTS.append({"lat": float(row['lat']), "lon": float(row['lon']), "name": str(row.get('facility', 'Facility'))})
        except: pass

def get_latest_log():
    try:
        logs = [f for f in LOG_DIR.glob("*.log") if f.is_file()]
        if not logs: return None
        logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return logs[0]
    except Exception: return None

def format_uptime(seconds):
    if seconds < 0: return "00:00:00"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def format_bytes(b):
    if b < 1024: return f"{b}B"
    elif b < 1024**2: return f"{b/1024:.1f}K"
    elif b < 1024**3: return f"{b/1024**2:.1f}M"
    else: return f"{b/1024**3:.1f}G"

def get_discovery_points():
    points = {}
    log_file = get_latest_log()
    if not log_file: return []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            current_loc_key = None
            for line in f:
                m_find = re.search(r"Finding samples within .* of \(([\d\.-]+),\s*([\d\.-]+)\)", line)
                if m_find:
                    lat, lon = float(m_find.group(1)), float(m_find.group(2))
                    current_loc_key = f"{lat},{lon}"
                    name = f"Site ({lat}, {lon})"
                    for fac in FACILITY_POINTS:
                        if abs(fac['lat'] - lat) < 0.05 and abs(fac['lon'] - lon) < 0.05:
                            name = fac['name']
                            break
                    points[current_loc_key] = {"lat": lat, "lon": lon, "count": "Scanning...", "name": name}
                m_found = re.search(r"Found (\d+) samples", line)
                if m_found and current_loc_key in points: points[current_loc_key]["count"] = m_found.group(1)
    except Exception: pass
    return list(points.values())

def load_sms_config():
    if SMS_CONFIG_FILE.exists():
        try:
            with open(SMS_CONFIG_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"enabled": False, "number": "", "carrier": "vtext.com", "level": "ERROR", "smtp_email": "", "smtp_password": ""}

# ==========================================
# 🟢 INITIALIZE GLOBAL STATE
# ==========================================
load_facilities()
SMS_STATE = load_sms_config()

# ==========================================
# 🟢 BACKGROUND THREADS
# ==========================================
def send_text_alert(level, message):
    global LAST_SMS_TIME
    if not SMS_STATE.get("enabled") or not SMS_STATE.get("number"): return
    
    if time.time() - LAST_SMS_TIME < 60: return
    LAST_SMS_TIME = time.time()

    gateway = SMS_STATE.get("carrier", "vtext.com")
    target_email = f"{SMS_STATE['number']}@{gateway}"
    sender_email = SMS_STATE.get("smtp_email")
    sender_pass = SMS_STATE.get("smtp_password")
    
    print(f"\n📱 [SMS TRIGGERED] -> {target_email}: {level} - {message[:50]}...\n")

    if sender_email and sender_pass:
        try:
            msg = EmailMessage()
            msg.set_content(f"AmpliScout [{level}]: {message[:100]}") 
            msg['Subject'] = "Pipeline Alert"
            msg['From'] = sender_email
            msg['To'] = target_email

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_pass)
                server.send_message(msg)
            print("✅ SMS dispatched successfully via SMTP.")
        except Exception as e:
            print(f"❌ SMS Delivery failed: {e}")

def sms_watcher_thread():
    level_hierarchy = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    last_pos = 0
    cur_log = None
    
    while True:
        if not SMS_STATE.get("enabled"):
            time.sleep(5)
            continue
            
        latest = get_latest_log()
        if latest and cur_log != latest:
            cur_log = latest
            last_pos = os.path.getsize(cur_log) 
            
        if cur_log:
            try:
                with open(cur_log, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        
                        m = re.search(r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s+([A-Z]+)\s+(.*)", line)
                        if m:
                            lvl, msg = m.group(1), m.group(2)
                            user_lvl = SMS_STATE.get("level", "ERROR")
                            if level_hierarchy.get(lvl, 0) >= level_hierarchy.get(user_lvl, 3):
                                send_text_alert(lvl, msg)
                    last_pos = f.tell()
            except: pass
        time.sleep(2)

threading.Thread(target=sms_watcher_thread, daemon=True).start()

def h5ad_scanner_thread():
    global H5AD_STATS
    try:
        import anndata as ad
    except ImportError:
        H5AD_STATS["status"] = "Error: 'anndata' missing. Run: pip install anndata"
        return
        
    parsed_files = {} 
    
    while True:
        try:
            current_points, current_comp = [], {}
            total_samples, file_count = 0, 0
            avail_cols = set()
            found_any = False
            last_error = None
            
            for d in ["02_qiime", "03_processed_data", "processed_data"]:
                p = PROJECT_ROOT / d
                if p.exists() and p.is_dir():
                    for file in p.rglob("*.h5ad"):
                        found_any = True
                        try:
                            mtime = file.stat().st_mtime
                            filepath = str(file)
                            
                            if filepath not in parsed_files or parsed_files[filepath]['mtime'] != mtime:
                                pts, comp, n_obs = [], {}, 0
                                cols_to_add = []
                                try:
                                    try:
                                        adata = ad.read_h5ad(file, backed='r')
                                    except Exception:
                                        # 🟢 OOM-KILLER SAFEGUARD: Do NOT load files > 250MB into full memory
                                        file_mb = file.stat().st_size / (1024 * 1024)
                                        if file_mb > 250:
                                            raise Exception(f"File too large ({file_mb:.1f}MB) to bypass compression")
                                        adata = ad.read_h5ad(file)
                                        
                                    n_obs = int(adata.n_obs)
                                    obs = adata.obs
                                    
                                    if not obs.empty:
                                        target_cols = [str(c) for c in obs.columns if str(c).startswith('env_') or str(c).startswith('empo_') or 'match' in str(c).lower()]
                                        cols_to_add = target_cols
                                        
                                        file_dists = {}
                                        for col in target_cols:
                                            if col in obs.columns:
                                                counts = obs[col].astype(str).value_counts().to_dict()
                                                file_dists[col] = {str(k): int(v) for k, v in counts.items()}
                                        
                                        comp_col = 'env_broad_scale' if 'env_broad_scale' in obs.columns else (target_cols[0] if target_cols else None)
                                        if comp_col:
                                            for k, v in obs[comp_col].astype(str).value_counts().items():
                                                comp[str(k)] = int(v)
                                                
                                        if 'lat' in obs.columns and 'lon' in obs.columns:
                                            subset = obs.drop_duplicates(subset=['lat', 'lon'])
                                            for _, row in subset.iterrows():
                                                lat_val = safe_float(row.get('lat'))
                                                lon_val = safe_float(row.get('lon'))
                                                
                                                if lat_val is not None and lon_val is not None:
                                                    title = str(row.get('study_title', str(row.get('description', 'No Title provided'))))
                                                    pts.append({
                                                        "lat": lat_val, "lon": lon_val,
                                                        "acc": str(row.get('study_accession', file.stem)),
                                                        "title": title, "n_samples": n_obs, "dists": file_dists
                                                    })
                                    parsed_files[filepath] = {'mtime': mtime, 'points': pts, 'comp': comp, 'samples': n_obs, 'cols': cols_to_add}
                                except Exception as e:
                                    print(f"❌ Failed to parse {file.name}: {e}")
                                    last_error = f"Error on {file.name}: {str(e)[:40]}"
                                    continue 
                            
                            if filepath in parsed_files:
                                fdata = parsed_files[filepath]
                                current_points.extend(fdata['points'])
                                for k, v in fdata['comp'].items(): current_comp[k] = current_comp.get(k, 0) + v
                                avail_cols.update(fdata.get('cols', []))
                                total_samples += fdata['samples']
                                file_count += 1
                                
                        except Exception: pass
            
            H5AD_STATS["total_files"] = file_count
            H5AD_STATS["total_samples"] = total_samples
            H5AD_STATS["points"] = current_points
            H5AD_STATS["metadata_composition"] = current_comp
            H5AD_STATS["available_columns"] = sorted(list(avail_cols))
            
            if not found_any:
                H5AD_STATS["status"] = "Scanning for files..."
            elif last_error:
                H5AD_STATS["status"] = last_error
            else:
                H5AD_STATS["status"] = "Active"
                
        except Exception as e: 
            print(f"❌ Scanner thread crashed: {e}")
            H5AD_STATS["status"] = f"Thread Crash: {str(e)[:40]}"
            
        time.sleep(10)

threading.Thread(target=h5ad_scanner_thread, daemon=True).start()

# ==========================================
# 🟢 FLASK ROUTES & UI
# ==========================================

MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AmpliScout Workflow Monitor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { background: #0f172a; color: #cbd5e1; font-family: 'Inter', sans-serif; height: 100vh; overflow: hidden; display: flex; flex-direction: column; font-size: 14px; }
        
        ::-webkit-scrollbar { width: 14px; height: 14px; }
        ::-webkit-scrollbar-track { background: #0f172a; border-left: 1px solid #1e293b; border-top: 1px solid #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 7px; border: 3px solid #0f172a; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        
        nav { height: 70px; flex-shrink: 0; background: #020617; border-bottom: 1px solid #1e293b; display: flex; align-items: center; justify-content: space-between; padding: 0 2rem; }
        main { display: flex; flex-grow: 1; overflow: hidden; }
        #workspace { flex-grow: 1; overflow-y: auto; padding: 1.5rem; position: relative; }
        
        #sidebar-logs { width: 600px; min-width: 400px; max-width: 50%; background: #020617; display: flex; flex-direction: column; border-left: 1px solid #1e293b; resize: horizontal; direction: rtl; overflow: hidden; }
        #sidebar-inner { direction: ltr; display: flex; flex-direction: column; height: 100%; width: 100%; }
        #log-terminal { font-family: 'Fira Code', monospace; font-size: 13px; flex-grow: 1; overflow-y: auto; padding: 1rem 1.5rem; color: #a3e635; line-height: 1.6; scroll-behavior: smooth; }

        .card { background: #1e293b; border-radius: 8px; padding: 1.25rem; border: 1px solid #334155; margin-bottom: 1.5rem; display: flex; flex-direction: column; transition: all 0.3s ease; }
        .card-header { font-size: 0.95rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;}
        
        .progress-bg { background: #334155; border-radius: 999px; height: 10px; width: 100%; overflow: hidden; margin-top: 6px; }
        .progress-fill { background: #3b82f6; height: 100%; transition: width 0.4s ease; }
        
        .timer-badge { background: #0f172a; padding: 0.35rem 0.85rem; border-radius: 6px; border: 1px solid #334155; display: flex; flex-direction: column; align-items: flex-end;}
        .timer-label { font-size: 0.70rem; font-weight: 800; color: #64748b; text-transform: uppercase; }
        .timer-value { font-family: 'Fira Code', monospace; font-size: 1rem; font-weight: 700; color: #f8fafc;}
        
        .htop-font { font-family: 'Fira Code', monospace; }
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 99999; justify-content: center; align-items: center; }
        .modal-content { background: #1e293b; border: 1px solid #334155; border-radius: 8px; width: 90%; max-width: 1000px; padding: 2rem; height: 90vh; display: flex; flex-direction: column;}
        
        .leaflet-container { border-radius: 6px !important; }
        .leaflet-popup-content-wrapper { background: #0f172a !important; color: #f8fafc !important; border: 1px solid #475569 !important; border-radius: 8px !important; width: 380px !important; box-shadow: 0 10px 25px rgba(0,0,0,0.8) !important; }
        .leaflet-popup-tip { background: #0f172a !important; border: 1px solid #475569 !important; }
        .leaflet-popup-content { margin: 16px !important; line-height: 1.5 !important; font-size: 14px !important; }
        .map-expanded { position: fixed !important; top: 1.5rem !important; left: 1.5rem !important; right: 1.5rem !important; bottom: 1.5rem !important; height: calc(100vh - 3rem) !important; width: calc(100vw - 3rem) !important; z-index: 9999 !important; border: 4px solid #3b82f6; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8); }
        
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #334155; transition: .4s; border-radius: 24px; }
        .slider:before { position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #10b981; }
        input:checked + .slider:before { transform: translateX(20px); }
    </style>
</head>
<body>
    <div id="sys-error" class="hidden absolute top-0 w-full z-[99999] shadow-lg"></div>

    <nav>
        <div class="flex items-center gap-4">
            <i class="fa-solid fa-dna text-blue-500 text-2xl"></i>
            <span class="text-xl font-black text-slate-200 tracking-wider">AMPLISCOUT WORKFLOW</span>
        </div>
        <div class="flex gap-3 items-center">
            <div class="timer-badge">
                <span class="timer-label">Server</span>
                <span class="timer-value text-slate-300" id="server-timer">00:00:00</span>
            </div>
            <div class="timer-badge">
                <span class="timer-label">Compute</span>
                <span class="timer-value text-blue-400" id="compute-timer">00:00:00</span>
            </div>
            <button onclick="openSmsModal()" class="ml-2 px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded text-sm font-bold transition-colors border border-slate-600 shadow-sm"><i class="fa-solid fa-bell text-amber-400 mr-2"></i> Alerts</button>
            <button onclick="window.open('/config', '_blank')" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-bold transition-colors shadow-lg"><i class="fa-solid fa-sliders mr-2"></i> Config</button>
        </div>
    </nav>

    <main>
        <div id="workspace">
            <div class="grid grid-cols-2 gap-6 mb-6">
                <div class="card cursor-pointer hover:border-slate-400 transition-colors relative" onclick="openChartModal()">
                    <div class="card-header">
                        <span><i class="fa-solid fa-server mr-2 text-blue-400"></i> Process Tree <span class="text-xs text-slate-500 font-normal ml-2">(Click for Telemetry)</span></span>
                        <span id="proc-status" class="text-slate-500 font-bold"><i class="fa-solid fa-circle-stop text-sm mr-1"></i> Offline</span>
                    </div>
                    
                    <div class="flex flex-col gap-3 mb-4">
                        <div class="flex items-start gap-4">
                            <div class="w-1/2">
                                <div class="text-xs font-bold text-slate-500 mb-2">CPU CORES</div>
                                <div id="core-grid" class="flex flex-wrap gap-[3px]"></div>
                            </div>
                            <div class="w-1/2 htop-font text-xs text-slate-300 bg-slate-900 p-3 rounded border border-slate-700 leading-relaxed" id="mem-status"></div>
                        </div>
                    </div>
                    
                    <div class="overflow-y-auto flex-grow max-h-56">
                        <table class="w-full text-left text-xs font-mono">
                            <thead><tr class="text-slate-500 border-b border-slate-700"><th class="pb-2 w-8"></th><th class="pb-2">PID</th><th class="pb-2">Process</th><th class="pb-2">S</th><th class="pb-2">THR</th><th class="pb-2">VIRT</th><th class="pb-2">RES</th><th class="pb-2">CPU%</th><th class="pb-2">RAM%</th></tr></thead>
                            <tbody id="htop-body"></tbody>
                        </table>
                    </div>
                </div>

                <div class="card flex flex-col">
                    <div class="card-header">
                        <span><i class="fa-solid fa-folder-tree mr-2 text-emerald-500"></i> Recent File Activity</span>
                        <div class="flex gap-2 text-sm items-center">
                            <button onclick="changePage(-1); event.stopPropagation();" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded shadow"><i class="fa-solid fa-chevron-left"></i></button>
                            <span id="page-indicator" class="px-2 text-slate-300 font-bold font-mono">1 / 1</span>
                            <button onclick="changePage(1); event.stopPropagation();" class="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-white rounded shadow"><i class="fa-solid fa-chevron-right"></i></button>
                        </div>
                    </div>
                    <ul id="file-activity" class="text-sm font-mono text-slate-400 space-y-3 overflow-y-auto flex-grow p-2"></ul>
                </div>
            </div>

            <div class="grid grid-cols-1 2xl:grid-cols-3 gap-6 mb-6">
                <div class="card col-span-1 flex flex-col" id="card-map-disc">
                    <div class="card-header flex flex-col xl:flex-row gap-3 items-start xl:items-center">
                        <span class="whitespace-nowrap"><i class="fa-solid fa-satellite-dish mr-2 text-blue-400"></i> Discovery Radar</span>
                        <div class="flex items-center gap-3 flex-wrap justify-end w-full xl:w-auto">
                            <span id="discovery-counter" class="whitespace-nowrap text-blue-400 font-bold text-sm bg-blue-900/30 px-2 py-1 rounded border border-blue-800 flex-shrink-0">0 Sites</span>
                            <button onclick="toggleMap('card-map-disc', mapDiscovery)" class="flex-shrink-0 text-slate-400 hover:text-white bg-slate-800 p-1.5 rounded" title="Toggle Fullscreen"><i class="fa-solid fa-expand"></i></button>
                        </div>
                    </div>
                    <div style="border-radius: 6px; overflow: hidden; height: 350px; position: relative;" class="w-full flex-grow border border-slate-700">
                        <div id="map-discovery" style="height: 100%; width: 100%; z-index: 1;" class="bg-slate-900"></div>
                    </div>
                </div>
                
                <div class="card col-span-1 flex flex-col" id="card-map-proc">
                    <div class="card-header flex flex-col xl:flex-row gap-3 items-start xl:items-center w-full">
                        <span class="whitespace-nowrap"><i class="fa-solid fa-map-location-dot mr-2 text-emerald-400"></i> Processed Data</span>
                        <div class="flex items-center gap-3 flex-wrap justify-start xl:justify-end w-full xl:w-auto ml-auto">
                            <span id="map-loading" class="whitespace-nowrap text-xs text-amber-400 font-bold hidden bg-amber-900/30 px-2 py-1 rounded"><i class="fa-solid fa-spinner fa-spin"></i> Updating...</span>
                            <label class="whitespace-nowrap text-xs flex items-center gap-1 cursor-pointer bg-slate-800 p-1.5 rounded hover:bg-slate-700 transition" title="Show facilities">
                                <input type="checkbox" id="toggle-facilities" onchange="toggleFacilities()" class="cursor-pointer" checked> <i class="fa-solid fa-building text-red-500"></i> Sites
                            </label>
                            <select id="map-color-by" class="bg-slate-800 text-xs text-slate-200 font-bold rounded border border-slate-600 p-1.5 outline-none cursor-pointer max-w-[250px]">
                                <option value="default">Color: Awaiting Data</option>
                            </select>
                            <span id="h5ad-counter" class="whitespace-nowrap flex-shrink-0 text-emerald-400 font-bold text-sm bg-emerald-900/30 px-2 py-1 rounded border border-emerald-800">0 Files</span>
                            <button onclick="toggleMap('card-map-proc', mapProcessed)" class="flex-shrink-0 text-slate-400 hover:text-white bg-slate-800 p-1.5 rounded" title="Toggle Fullscreen"><i class="fa-solid fa-expand"></i></button>
                        </div>
                    </div>
                    <div style="border-radius: 6px; overflow: hidden; height: 350px; position: relative;" class="w-full flex-grow border border-slate-700">
                        <div id="map-processed" style="height: 100%; width: 100%; z-index: 1;" class="bg-slate-900"></div>
                    </div>
                </div>
                
                <div class="card col-span-1 flex flex-col">
                    <div class="card-header">
                        <span><i class="fa-solid fa-chart-pie mr-2 text-purple-400"></i> Composition</span>
                    </div>
                    <div style="height: 350px; width: 100%; position: relative; flex-grow: 1;"><canvas id="metaChart"></canvas></div>
                </div>
            </div>

            <div class="card" id="task-container">
                <div class="card-header"><span><i class="fa-solid fa-bars-progress mr-2 text-blue-500"></i> Pipeline Execution</span></div>
                <div id="progress-bars" class="space-y-6 text-base text-slate-500 font-medium italic p-2">Awaiting telemetry sync...</div>
            </div>
        </div>

        <div id="sidebar-logs">
            <div id="sidebar-inner">
                <div class="p-4 border-b border-slate-800 bg-slate-950 flex justify-between items-center shadow-lg z-10">
                    <span class="text-sm font-bold text-slate-300 uppercase tracking-widest"><i class="fa-solid fa-terminal mr-2 text-emerald-500"></i> Live Log Stream</span>
                </div>
                <div id="log-terminal"></div>
            </div>
        </div>
    </main>

    <div id="chartModal" class="modal">
        <div class="modal-content">
            <div class="flex justify-between items-center mb-6 text-slate-200">
                <h2 class="font-bold text-xl uppercase tracking-wider"><i class="fa-solid fa-server mr-2 text-blue-500"></i> System Telemetry (60s)</h2>
                <button onclick="closeChartModal(); event.stopPropagation();" class="text-slate-400 hover:text-white text-3xl"><i class="fa-solid fa-xmark"></i></button>
            </div>
            <div class="flex-grow flex flex-col gap-8 h-full">
                <div class="h-1/2 w-full relative"><canvas id="cpuChart"></canvas></div>
                <div class="h-1/2 w-full relative"><canvas id="ramChart"></canvas></div>
            </div>
        </div>
    </div>

    <div id="smsModal" class="modal" style="align-items: flex-start; padding-top: 5rem;">
        <div class="modal-content" style="max-width: 500px; height: auto;">
            <div class="flex justify-between items-center mb-4 border-b border-slate-700 pb-4">
                <h2 class="font-bold text-xl text-white"><i class="fa-solid fa-bell text-amber-400 mr-2"></i> Pipeline Alerts</h2>
                <button onclick="closeSmsModal()" class="text-slate-400 hover:text-white text-xl"><i class="fa-solid fa-xmark"></i></button>
            </div>
            
            <div class="space-y-5">
                <div class="flex justify-between items-center bg-slate-900 p-4 rounded border border-slate-700">
                    <div>
                        <div class="font-bold text-white mb-1">Enable SMS Notifications</div>
                        <div class="text-xs text-slate-400">Receive texts when the pipeline hits a specific log level.</div>
                    </div>
                    <label class="switch">
                        <input type="checkbox" id="sms-enable">
                        <span class="slider"></span>
                    </label>
                </div>
                
                <div>
                    <label class="block text-xs font-bold text-slate-400 uppercase mb-2">Phone Number</label>
                    <input type="text" id="sms-number" placeholder="e.g. 5551234567" class="w-full bg-slate-900 border border-slate-700 rounded p-2.5 text-white focus:outline-none focus:border-blue-500 font-mono">
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-bold text-slate-400 uppercase mb-2">Carrier Gateway</label>
                        <select id="sms-carrier" class="w-full bg-slate-900 border border-slate-700 rounded p-2.5 text-white focus:outline-none focus:border-blue-500 cursor-pointer">
                            <option value="vtext.com">Verizon</option>
                            <option value="txt.att.net">AT&T</option>
                            <option value="tmomail.net">T-Mobile / Mint</option>
                            <option value="messaging.sprintpcs.com">Sprint</option>
                            <option value="msg.fi.google.com">Google Fi</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs font-bold text-slate-400 uppercase mb-2">Minimum Log Level</label>
                        <select id="sms-level" class="w-full bg-slate-900 border border-slate-700 rounded p-2.5 text-white focus:outline-none focus:border-blue-500 cursor-pointer">
                            <option value="INFO">INFO (All updates)</option>
                            <option value="WARNING">WARNING</option>
                            <option value="ERROR" selected>ERROR & CRITICAL</option>
                        </select>
                    </div>
                </div>
                
                <div class="mt-4 pt-4 border-t border-slate-700">
                    <h3 class="text-xs font-bold text-slate-400 uppercase mb-3"><i class="fa-solid fa-envelope mr-1"></i> SMTP Sender Configuration</h3>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">Gmail Address</label>
                            <input type="text" id="sms-email" placeholder="alert-bot@gmail.com" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-sm focus:border-blue-500 outline-none">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-400 mb-1">App Password</label>
                            <input type="password" id="sms-password" placeholder="16-char-password" class="w-full bg-slate-900 border border-slate-700 rounded p-2 text-white text-sm focus:border-blue-500 outline-none">
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-6 flex justify-end gap-3">
                <span id="sms-status" class="text-emerald-400 font-bold self-center hidden text-sm"><i class="fa-solid fa-check mr-1"></i> Saved</span>
                <button onclick="saveSmsConfig()" class="bg-blue-600 hover:bg-blue-500 text-white font-bold py-2 px-6 rounded transition">Save Preferences</button>
            </div>
        </div>
    </div>

    <script>
        let allFiles = [];
        let currentPage = 0;
        const PAGE_SIZE = 8;
        let isTreeExpanded = false;
        
        let processedMarkers = [];
        let facilityLayer = null;
        let rawFacilityData = [];

        function stringToColor(str) {
            let hash = 0;
            for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
            const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
            return '#' + '00000'.substring(0, 6 - c.length) + c;
        }

        function createStackedBar(distObj) {
            if (!distObj || Object.keys(distObj).length === 0) return '<div class="text-xs text-slate-500 italic">No category data</div>';
            let total = Object.values(distObj).reduce((a,b)=>a+b, 0);
            let html = '<div class="flex w-full h-5 rounded overflow-hidden mt-3 shadow-inner border border-slate-700">';
            let keys = Object.keys(distObj).sort((a,b) => distObj[b] - distObj[a]);
            
            keys.forEach(k => {
                let pct = (distObj[k] / total) * 100;
                let color = stringToColor(k);
                html += `<div style="width:${pct}%; background-color:${color};" title="${k}: ${distObj[k]} (${pct.toFixed(1)}%)"></div>`;
            });
            html += '</div><div class="flex flex-wrap gap-1 mt-2">';
            
            keys.slice(0, 4).forEach(k => {
                let color = stringToColor(k);
                html += `<span class="text-[10px] px-1.5 py-0.5 rounded text-white font-bold truncate max-w-[100px] drop-shadow-md" style="background-color:${color}; text-shadow: 0 1px 2px rgba(0,0,0,0.8);" title="${k}">${k}</span>`;
            });
            if(keys.length > 4) html += `<span class="text-[10px] text-slate-400 px-1 py-0.5 bg-slate-800 rounded font-bold border border-slate-600">+${keys.length-4} more</span>`;
            html += '</div>';
            return html;
        }

        function toggleTree(e) {
            e.stopPropagation();
            isTreeExpanded = !isTreeExpanded;
            const childProcs = document.querySelectorAll('.child-proc');
            if (childProcs) {
                childProcs.forEach(el => el.style.display = isTreeExpanded ? 'table-row' : 'none');
            }
            const treeBtn = document.getElementById('tree-btn');
            if (treeBtn) treeBtn.innerHTML = isTreeExpanded ? '<i class="fa-solid fa-minus"></i>' : '<i class="fa-solid fa-plus"></i>';
        }

        function changePage(dir) {
            const maxPage = Math.ceil(allFiles.length / PAGE_SIZE) - 1;
            currentPage += dir;
            if(currentPage < 0) currentPage = 0;
            if(currentPage > maxPage) currentPage = maxPage;
            renderFiles();
        }

        function renderFiles() {
            const fileEl = document.getElementById('file-activity');
            const pageEl = document.getElementById('page-indicator');
            if(!fileEl || !pageEl) return;
            
            if(!allFiles.length) {
                fileEl.innerHTML = '<li class="text-slate-500 italic">Monitoring directories...</li>';
                pageEl.innerText = '0 / 0';
                return;
            }
            const start = currentPage * PAGE_SIZE;
            const slice = allFiles.slice(start, start + PAGE_SIZE);
            fileEl.innerHTML = slice.map(f => `
                <li class="flex items-center gap-3 bg-slate-800/50 p-2 rounded border border-slate-700/50 hover:border-emerald-500/50 transition text-slate-300">
                    <span class="text-emerald-500 font-black text-xl leading-none">+</span> ${f}
                </li>
            `).join('');
            pageEl.innerText = `${currentPage + 1} / ${Math.max(1, Math.ceil(allFiles.length / PAGE_SIZE))}`;
        }

        let cpuChart, ramChart, metaChart;
        function initCharts() {
            const commonOptions = { responsive: true, maintainAspectRatio: false, animation: false, elements: { point: { radius: 0, hoverRadius: 6 } }, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: true, labels: { color: '#cbd5e1', font: {size: 14} } } } };
            
            if (document.getElementById('cpuChart')) {
                cpuChart = new Chart(document.getElementById('cpuChart').getContext('2d'), {
                    type: 'line', data: { labels: [], datasets: [{ label: 'Global CPU Usage (%)', borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', data: [], fill: true, tension: 0.2 }] },
                    options: { ...commonOptions, scales: { x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } }, y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } } } }
                });
            }

            if (document.getElementById('ramChart')) {
                ramChart = new Chart(document.getElementById('ramChart').getContext('2d'), {
                    type: 'line', data: { labels: [], datasets: [{ label: 'Global RAM Usage (%)', borderColor: '#10b981', backgroundColor: 'rgba(16, 185, 129, 0.1)', data: [], fill: true, tension: 0.2 }] },
                    options: { ...commonOptions, scales: { x: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } }, y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: '#1e293b' } } } }
                });
            }

            if (document.getElementById('metaChart')) {
                metaChart = new Chart(document.getElementById('metaChart').getContext('2d'), {
                    type: 'doughnut', data: { labels: [], datasets: [{ data: [], backgroundColor: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4'], borderWidth: 0 }] },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#cbd5e1', font: {size: 12} } } }, cutout: '75%' }
                });
            }
        }
        initCharts();

        function openChartModal() { 
            const m = document.getElementById('chartModal');
            if (m) m.style.display = 'flex'; 
        }
        function closeChartModal() { 
            const m = document.getElementById('chartModal');
            if (m) m.style.display = 'none'; 
        }
        
        async function openSmsModal() {
            const m = document.getElementById('smsModal');
            if (!m) return;
            m.style.display = 'flex';
            try {
                const res = await fetch('/api/sms_config');
                if(res.ok) {
                    const data = await res.json();
                    document.getElementById('sms-enable').checked = data.enabled;
                    document.getElementById('sms-number').value = data.number;
                    document.getElementById('sms-carrier').value = data.carrier;
                    document.getElementById('sms-level').value = data.level;
                    document.getElementById('sms-email').value = data.smtp_email || "";
                    document.getElementById('sms-password').value = data.smtp_password || "";
                }
            } catch(e) {}
        }
        function closeSmsModal() { 
            const m = document.getElementById('smsModal');
            if (m) m.style.display = 'none'; 
        }
        
        async function saveSmsConfig() {
            const payload = {
                enabled: document.getElementById('sms-enable').checked,
                number: document.getElementById('sms-number').value,
                carrier: document.getElementById('sms-carrier').value,
                level: document.getElementById('sms-level').value,
                smtp_email: document.getElementById('sms-email').value,
                smtp_password: document.getElementById('sms-password').value
            };
            try {
                const res = await fetch('/api/sms_config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                if(res.ok) {
                    const s = document.getElementById('sms-status');
                    if (s) {
                        s.style.display = 'block';
                        setTimeout(() => s.style.display = 'none', 2000);
                    }
                }
            } catch(e) {}
        }

        const mapDiscovery = L.map('map-discovery', { preferCanvas: true }).setView([20, 0], 1);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', { attribution: '&copy; Esri' }).addTo(mapDiscovery);
        let markersDiscovery = L.layerGroup().addTo(mapDiscovery);

        const mapProcessed = L.map('map-processed', { preferCanvas: true }).setView([20, 0], 1);
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', { attribution: '&copy; Esri' }).addTo(mapProcessed);
        let markersProcessedGroup = L.layerGroup().addTo(mapProcessed);
        facilityLayer = L.layerGroup().addTo(mapProcessed);

        function toggleMap(cardId, mapObj) {
            const card = document.getElementById(cardId);
            if (card) {
                card.classList.toggle('map-expanded');
                setTimeout(() => mapObj.invalidateSize(true), 300);
            }
        }

        function toggleFacilities() {
            const cb = document.getElementById('toggle-facilities');
            if (!cb) return;
            
            facilityLayer.clearLayers();
            if(cb.checked) {
                rawFacilityData.forEach(pt => {
                    const marker = L.marker([pt.lat, pt.lon], {
                        icon: L.divIcon({ className: 'bg-transparent', html: '<i class="fa-solid fa-play fa-rotate-270 text-red-500 text-[10px] drop-shadow-md"></i>', iconSize: [12, 12], iconAnchor: [6, 6] })
                    }).bindTooltip(`<b class="text-red-400">${pt.name}</b>`, {direction: 'top', offset: [0, -8]});
                    facilityLayer.addLayer(marker);
                });
            }
        }

        const colorSelect = document.getElementById('map-color-by');
        if (colorSelect) {
            colorSelect.addEventListener('change', (e) => {
                const loader = document.getElementById('map-loading');
                if (loader) loader.classList.remove('hidden');
                
                const key = e.target.value;
                
                setTimeout(() => {
                    processedMarkers.forEach(item => {
                        let val = "Unknown";
                        if(key === 'acc') val = item.data.acc;
                        else if(item.data.dists && item.data.dists[key]) {
                            const dist = item.data.dists[key];
                            if (Object.keys(dist).length > 0) {
                                val = Object.keys(dist).reduce((a, b) => dist[a] > dist[b] ? a : b);
                            }
                        }
                        item.marker.setStyle({ color: stringToColor(val) });
                        
                        const popupHTML = `
                            <div class="text-sm font-sans">
                                <b class="text-emerald-400 text-xl block mb-1 tracking-tight">${item.data.acc} <span class="text-emerald-300 font-mono text-sm float-right bg-slate-800 px-2 py-1 rounded border border-slate-600">N=${item.data.n_samples}</span></b>
                                <span class="text-slate-300 italic block mb-3 border-b border-slate-600 pb-3 leading-relaxed">${item.data.title}</span>
                                <div class="bg-slate-900 p-3 rounded border border-slate-700">
                                    <span class="text-slate-400 font-bold text-[10px] uppercase tracking-wider mb-1 block"><i class="fa-solid fa-chart-column mr-1 text-slate-500"></i> Distribution: ${key.replace(/_/g, ' ').toUpperCase()}</span>
                                    ${createStackedBar(item.data.dists[key] || {})}
                                </div>
                            </div>
                        `;
                        item.marker.getPopup().setContent(popupHTML);
                    });
                    if (loader) loader.classList.add('hidden');
                }, 50);
            });
        }

        // --- MAIN POLLING LOOP WITH BUILT-IN DIAGNOSTICS ---
        setInterval(async () => {
            try {
                const res = await fetch('/monitor_data');
                const text = await res.text();
                
                let data;
                try {
                    data = JSON.parse(text);
                } catch(e) {
                    throw new Error(`Corrupt JSON received from backend.\nRaw Response: ${text.substring(0, 300)}`);
                }

                if (!res.ok) throw new Error(`HTTP ${res.status} Backend Error:\n${data.error || "Unknown Server Error"}`);
                if (data.error) throw new Error(`Python Exception Caught:\n${data.error}`);

                const errBanner = document.getElementById('sys-error');
                if (errBanner) errBanner.style.display = 'none';
                
                const elServerTimer = document.getElementById('server-timer');
                if (elServerTimer) elServerTimer.innerText = data.server_uptime;
                
                const elComputeTimer = document.getElementById('compute-timer');
                const elProcStatus = document.getElementById('proc-status');
                
                if (data.is_running) {
                    if (elComputeTimer) {
                        elComputeTimer.innerText = data.compute_uptime;
                        elComputeTimer.classList.add("text-blue-400");
                    }
                    if (elProcStatus) elProcStatus.innerHTML = '<i class="fa-solid fa-circle-play text-sm mr-1 text-emerald-400 shadow"></i> Running';
                } else {
                    if (elComputeTimer) {
                        elComputeTimer.innerText = "OFFLINE";
                        elComputeTimer.classList.remove("text-blue-400");
                    }
                    if (elProcStatus) elProcStatus.innerHTML = '<i class="fa-solid fa-circle-stop text-sm mr-1 text-slate-500"></i> Offline';
                }

                if(data.htop) {
                    const elCoreGrid = document.getElementById('core-grid');
                    if (elCoreGrid) {
                        elCoreGrid.innerHTML = data.htop.cores.map(pct => {
                            let colorStr = '#1e293b'; 
                            if (pct > 0.5) { 
                                let hue = Math.max(0, 120 - (pct * 1.2)); 
                                colorStr = `hsl(${hue}, 80%, 50%)`; 
                            }
                            return `<div title="${pct.toFixed(1)}%" class="w-3 h-3 md:w-3.5 md:h-3.5 rounded-[2px] border border-slate-900 shadow-sm" style="background-color: ${colorStr}"></div>`;
                        }).join('');
                    }

                    const elMemStatus = document.getElementById('mem-status');
                    if (elMemStatus) {
                        elMemStatus.innerHTML = `
                            Load avg: <span class="text-white">${data.htop.load}</span><br>
                            ${data.htop.mem_str}<br>
                            ${data.htop.swp_str}<br>
                            Tasks: <b class="text-blue-400">${data.htop.tasks}</b> | Uptime: ${data.htop.uptime}
                        `;
                    }
                }

                const elHtopBody = document.getElementById('htop-body');
                if (elHtopBody) {
                    if (data.processes.length > 0) {
                        const master = data.processes[0];
                        const hasChildren = data.processes.length > 1;
                        
                        const toggleBtn = hasChildren 
                            ? `<button id="tree-btn" class="text-slate-300 hover:text-white bg-slate-600 px-2 py-0.5 rounded shadow" onclick="toggleTree(event)">${isTreeExpanded ? '<i class="fa-solid fa-minus"></i>' : '<i class="fa-solid fa-plus"></i>'}</button>`
                            : `<span class="text-slate-600 px-2"><i class="fa-solid fa-circle-dot text-[8px]"></i></span>`;

                        let html = `
                            <tr class="border-b border-slate-700 bg-slate-800/80">
                                <td class="py-2.5 text-center">${toggleBtn}</td>
                                <td class="py-2.5 text-slate-300">${master.pid}</td>
                                <td class="py-2.5 font-bold text-blue-400">${master.name}</td>
                                <td class="py-2.5 font-bold ${master.status === 'R' ? 'text-emerald-400' : 'text-amber-400'}">${master.status}</td>
                                <td class="py-2.5 text-slate-300">${master.threads}</td>
                                <td class="py-2.5 text-slate-400">${master.virt}</td>
                                <td class="py-2.5 text-slate-400">${master.res}</td>
                                <td class="py-2.5 text-emerald-400 font-bold">${master.cpu.toFixed(1)}%</td>
                                <td class="py-2.5 text-amber-400 font-bold">${master.ram.toFixed(1)}%</td>
                            </tr>
                        `;
                        for(let i=1; i<data.processes.length; i++) {
                            const p = data.processes[i];
                            html += `
                            <tr class="border-b border-slate-800/50 child-proc bg-slate-900" style="display: ${isTreeExpanded ? 'table-row' : 'none'};">
                                <td class="py-1.5 text-center text-slate-600">↳</td>
                                <td class="py-1.5 text-slate-500">${p.pid}</td>
                                <td class="py-1.5 text-slate-400">${p.name}</td>
                                <td class="py-1.5 font-bold ${p.status === 'R' ? 'text-emerald-500' : 'text-slate-600'}">${p.status}</td>
                                <td class="py-1.5 text-slate-500">${p.threads}</td>
                                <td class="py-1.5 text-slate-500">${p.virt}</td>
                                <td class="py-1.5 text-slate-500">${p.res}</td>
                                <td class="py-1.5 text-emerald-500">${p.cpu.toFixed(1)}%</td>
                                <td class="py-1.5 text-amber-500">${p.ram.toFixed(1)}%</td>
                            </tr>`;
                        }
                        elHtopBody.innerHTML = html;
                    } else {
                        elHtopBody.innerHTML = `<tr><td colspan="9" class="py-4 text-center text-slate-500 italic">No AmpliScout processes detected</td></tr>`;
                    }
                }

                allFiles = data.recent_files;
                renderFiles();
                
                if(data.state && data.state.tasks) {
                    const elProgress = document.getElementById('progress-bars');
                    if (elProgress) {
                        elProgress.innerHTML = data.state.tasks.map(t => `
                            <div class="mb-5 last:mb-0 bg-slate-800/30 p-4 rounded border border-slate-700">
                                <div class="flex justify-between text-sm uppercase tracking-wider font-bold text-slate-200 mb-2">
                                    <span>${t.description.replace(/\[.*?\]/g, '')}</span>
                                    <span class="text-slate-400 bg-slate-900 px-3 py-1 rounded border border-slate-700 shadow-inner">${t.completed} / ${t.total} <span class="text-blue-400 ml-2 font-black">(${t.percentage}%)</span></span>
                                </div>
                                <div class="progress-bg"><div class="progress-fill shadow-[0_0_10px_rgba(59,130,246,0.6)]" style="width: ${t.percentage}%"></div></div>
                            </div>
                        `).join('');
                    }
                }

                if(data.sys_history) {
                    if (cpuChart) {
                        cpuChart.data.labels = data.sys_history.map(h => h.time);
                        cpuChart.data.datasets[0].data = data.sys_history.map(h => h.cpu);
                        cpuChart.update();
                    }
                    if (ramChart) {
                        ramChart.data.labels = data.sys_history.map(h => h.time);
                        ramChart.data.datasets[0].data = data.sys_history.map(h => h.ram);
                        ramChart.update();
                    }
                }

                if(data.discovery_points) {
                    const elDiscCounter = document.getElementById('discovery-counter');
                    if (elDiscCounter) elDiscCounter.innerText = `${data.discovery_points.length} Sites Scanned`;
                    
                    markersDiscovery.clearLayers();
                    data.discovery_points.forEach(pt => {
                        L.circleMarker([pt.lat, pt.lon], {radius: 7, color: '#3b82f6', fillColor: '#60a5fa', fillOpacity: 0.4, weight: 2})
                         .bindTooltip(`<div class="text-sm p-1"><b class="text-blue-400 border-b border-slate-600 pb-1 block mb-2">${pt.name}</b><span class="text-slate-300">Hits: <b class="text-emerald-400 text-lg">${pt.count}</b></span></div>`)
                         .addTo(markersDiscovery);
                    });
                }

                if(data.h5ad_stats) {
                    const elMapLoading = document.getElementById('map-loading');
                    if (elMapLoading && data.h5ad_stats.status) {
                        if (data.h5ad_stats.status.startsWith('Error') || data.h5ad_stats.status.startsWith('Warning') || data.h5ad_stats.status.startsWith('Missing') || data.h5ad_stats.status.startsWith('Thread')) {
                            elMapLoading.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-red-500"></i> <span class="text-red-400" title="${data.h5ad_stats.status}">${data.h5ad_stats.status.substring(0,30)}...</span>`;
                            elMapLoading.classList.remove('hidden');
                            elMapLoading.classList.remove('bg-amber-900/30');
                            elMapLoading.classList.add('bg-red-900/40');
                        } else if (data.h5ad_stats.status === 'Active') {
                            elMapLoading.classList.add('hidden');
                        } else {
                            elMapLoading.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span class="text-amber-400">${data.h5ad_stats.status}</span>`;
                            elMapLoading.classList.remove('hidden');
                        }
                    }
                
                    if (data.h5ad_stats.points.length !== processedMarkers.length) {
                        const elH5adCounter = document.getElementById('h5ad-counter');
                        if (elH5adCounter) elH5adCounter.innerText = `${data.h5ad_stats.total_files} Files | ${data.h5ad_stats.total_samples} Samples`;
                        
                        if (metaChart) {
                            const comp = data.h5ad_stats.metadata_composition;
                            metaChart.data.labels = Object.keys(comp);
                            metaChart.data.datasets[0].data = Object.values(comp);
                            metaChart.update();
                        }
                        
                        const select = document.getElementById('map-color-by');
                        if (select) {
                            const currVal = select.value;
                            let optsHTML = `<option value="acc">Color: Project Accession</option>`;
                            
                            if (data.h5ad_stats.available_columns && data.h5ad_stats.available_columns.length > 0) {
                                data.h5ad_stats.available_columns.forEach(col => {
                                    optsHTML += `<option value="${col}">Color: ${col.replace(/_/g, ' ').toUpperCase()}</option>`;
                                });
                            }
                            select.innerHTML = optsHTML;
                            
                            if(data.h5ad_stats.available_columns.includes(currVal)) {
                                select.value = currVal;
                            } else if (data.h5ad_stats.available_columns.includes('env_broad_scale')) {
                                select.value = 'env_broad_scale';
                            }

                            markersProcessedGroup.clearLayers();
                            processedMarkers = [];
                            const activeKey = select.value;
                            
                            data.h5ad_stats.points.forEach(pt => {
                                let colVal = "Unknown";
                                if(activeKey === 'acc') colVal = pt.acc;
                                else if(pt.dists && pt.dists[activeKey]) {
                                    const dist = pt.dists[activeKey];
                                    if (Object.keys(dist).length > 0) {
                                        colVal = Object.keys(dist).reduce((a, b) => dist[a] > dist[b] ? a : b);
                                    }
                                }
                                
                                const marker = L.circleMarker([pt.lat, pt.lon], {radius: 6, color: stringToColor(colVal), fillOpacity: 0.9, weight: 1});
                                
                                const popupHTML = `
                                    <div class="text-sm font-sans">
                                        <b class="text-emerald-400 text-xl block mb-1 tracking-tight">${pt.acc} <span class="text-emerald-300 font-mono text-sm float-right bg-slate-800 px-2 py-1 rounded border border-slate-600">N=${pt.n_samples}</span></b>
                                        <span class="text-slate-300 italic block mb-3 border-b border-slate-600 pb-3 leading-relaxed">${pt.title}</span>
                                        <div class="bg-slate-900 p-3 rounded border border-slate-700">
                                            <span class="text-slate-400 font-bold text-[10px] uppercase tracking-wider mb-1 block"><i class="fa-solid fa-chart-column mr-1 text-slate-500"></i> Distribution: ${activeKey.replace(/_/g, ' ').toUpperCase()}</span>
                                            ${createStackedBar(pt.dists[activeKey] || {})}
                                        </div>
                                    </div>
                                `;
                                marker.bindPopup(popupHTML).addTo(markersProcessedGroup);
                                processedMarkers.push({marker: marker, data: pt});
                            });
                        }
                    }
                }
                
                if (data.facilities.length !== rawFacilityData.length) {
                    rawFacilityData = data.facilities;
                    toggleFacilities();
                }

            } catch(err) {
                console.error(err);
                const errBanner = document.getElementById('sys-error');
                if (errBanner) {
                    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
                        errBanner.innerHTML = `<div class="flex items-center"><i class="fa-solid fa-plug-circle-xmark text-xl mr-3 text-amber-500"></i> <div><b class="text-amber-400 uppercase tracking-widest">Server Disconnected</b><br><span class="text-amber-100">Connection lost to Python backend. The script may have been manually stopped or killed by the OS.</span></div></div>`;
                        errBanner.className = "bg-amber-900/90 border-b-4 border-amber-600 text-amber-100 p-4 font-sans text-sm w-full z-[99999] absolute top-0 shadow-lg";
                    } else {
                        errBanner.innerText = `CRITICAL UI CRASH INTERCEPTED:\n${err.toString()}`;
                        errBanner.className = "bg-red-900/90 border-b-4 border-red-600 text-red-200 p-4 font-mono text-xs w-full overflow-x-auto whitespace-pre-wrap z-[99999] absolute top-0 shadow-lg";
                    }
                    errBanner.style.display = 'block';
                }
            } 
        }, 1000);

        // --- BEAUTIFUL LOG STREAM PARSER ---
        const logBox = document.getElementById('log-terminal');
        const source = new EventSource("/stream_logs");
        source.onmessage = (e) => {
            if (!logBox) return;
            let text = e.data;
            const regex = /^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+([A-Z]+)\s+(.*)/;
            const match = text.match(regex);
            
            if(match) {
                const ts = match[1];
                const lvl = match[2];
                const msg = match[3];
                
                let colorClass = "text-slate-400 bg-slate-800"; 
                if (lvl === 'INFO') colorClass = "text-blue-400 bg-blue-900/40";
                else if (lvl === 'WARNING') colorClass = "text-amber-400 bg-amber-900/50";
                else if (lvl === 'ERROR' || lvl === 'CRITICAL') colorClass = "text-red-400 bg-red-900/50";
                
                text = `<div class="flex items-start gap-3 mb-1.5 hover:bg-slate-800/30 rounded p-1 transition"><span class="text-slate-500 font-mono text-[10px] whitespace-nowrap mt-1">${ts}</span><span class="${colorClass} px-2 py-0.5 rounded text-[9px] font-black w-14 text-center shrink-0 mt-0.5">${lvl}</span><span class="flex-grow text-slate-300 leading-snug break-words">${msg}</span></div>`;
            } else {
                text = `<div class="text-slate-400 mb-1.5 ml-[120px] pl-4 border-l-2 border-slate-700 italic text-xs">${text}</div>`;
            }
            
            const isScrolledToBottom = logBox.scrollHeight - logBox.clientHeight <= logBox.scrollTop + 30;
            logBox.innerHTML += text;
            if (isScrolledToBottom) logBox.scrollTo({top: logBox.scrollHeight, behavior: 'smooth'});
        };
    </script>
</body>
</html>
"""

CONFIG_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Config Editor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style> body { background: #0f172a; color: #cbd5e1; font-family: 'Inter', sans-serif; padding: 2rem; } textarea { width: 100%; height: 80vh; background: #1e293b; color: #a3e635; font-family: monospace; font-size: 14px; padding: 1.5rem; border: 1px solid #334155; border-radius: 8px; } </style>
</head>
<body>
    <div class="flex justify-between items-center mb-4">
        <h1 class="text-2xl font-bold text-white">Edit config.yaml</h1>
        <button onclick="saveConfig()" class="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded font-bold text-sm">Save Changes</button>
    </div>
    <div id="status" class="mb-4 text-emerald-400 font-bold hidden">Config saved successfully!</div>
    <textarea id="editor">{{ content }}</textarea>
    <script>
        async function saveConfig() {
            const content = document.getElementById('editor').value;
            const res = await fetch('/save_config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content }) });
            if(res.ok) {
                const s = document.getElementById('status');
                s.style.display = 'block';
                setTimeout(() => s.style.display = 'none', 3000);
            } else alert("Failed to save. Invalid YAML.");
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(MAIN_HTML)

@app.route('/config')
def edit_config():
    content = ""
    try:
        with open(CONFIG_FILE, 'r') as f: content = f.read()
    except Exception: content = "# Could not load config.yaml"
    return render_template_string(CONFIG_HTML, content=content)

@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        data = request.json['content']
        yaml.safe_load(data)
        with open(CONFIG_FILE, 'w') as f: f.write(data)
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/sms_config', methods=['GET', 'POST'])
def handle_sms_config():
    global SMS_STATE
    if request.method == 'POST':
        data = request.json
        SMS_STATE.update(data)
        try:
            with open(SMS_CONFIG_FILE, 'w') as f: json.dump(SMS_STATE, f)
        except Exception: pass
        return jsonify({"status": "success"})
    return jsonify(SMS_STATE)

@app.route('/monitor_data')
def monitor_data():
    try:
        global PROC_CACHE
        server_uptime = time.time() - SERVER_START_TIME
        processes, compute_uptime_sec, is_running, total_cpu, total_ram = [], 0, False, 0.0, 0.0

        try:
            active_pids = set()
            for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
                try:
                    cmd_list = p.info.get('cmdline') or [] 
                    cmd_str = " ".join([str(x) for x in cmd_list]).lower()
                    
                    if 'workflow_16s' in cmd_str and 'config_server' not in cmd_str:
                        pid = p.pid
                        active_pids.add(pid)
                        if pid not in PROC_CACHE: 
                            PROC_CACHE[pid] = psutil.Process(pid) 
                except Exception: pass

            for pid in list(PROC_CACHE.keys()):
                if pid not in active_pids: del PROC_CACHE[pid]
                
            target_procs = list(PROC_CACHE.values())
            
            if target_procs:
                valid_procs = []
                for p in target_procs:
                    try:
                        ctime = p.create_time()
                        valid_procs.append((p, ctime))
                    except psutil.NoSuchProcess: pass
                
                valid_procs.sort(key=lambda x: x[1])
                
                if valid_procs:
                    is_running = True
                    master_proc = valid_procs[0][0]
                    compute_uptime_sec = time.time() - valid_procs[0][1]
                    
                    for p, _ in valid_procs:
                        try:
                            cpu = float(p.cpu_percent(interval=None))
                            ram = float(p.memory_percent())
                            total_cpu += cpu
                            total_ram += ram
                            
                            mem_info = p.memory_info()
                            vms = format_bytes(mem_info.vms) if mem_info else "0B"
                            rss = format_bytes(mem_info.rss) if mem_info else "0B"
                            status = p.status()[:1].upper()
                            threads = int(p.num_threads())

                            processes.append({
                                "pid": p.pid, 
                                "name": "AmpliScout Master" if p.pid == master_proc.pid else f"Worker Subprocess", 
                                "cpu": cpu, "ram": ram, "virt": vms, "res": rss, "status": status, "threads": threads
                            })
                        except Exception: pass
        except Exception: pass

        timestamp = datetime.now().strftime("%H:%M:%S")
        SYS_HISTORY.append({"time": timestamp, "cpu": total_cpu, "ram": total_ram})
        if len(SYS_HISTORY) > 60: SYS_HISTORY.pop(0)

        state = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f: state = json.load(f)
            except Exception: pass

        recent = []
        try:
            for d in ["01_raw_data", "02_qiime", "07_logs", "03_processed_data", "processed_data"]:
                p = PROJECT_ROOT / d
                if p.exists() and p.is_dir():
                    items = []
                    for file in p.iterdir():
                        if file.is_file():
                            try: items.append((f"{d}/{file.name}", float(file.stat().st_mtime)))
                            except Exception: pass
                    items.sort(key=lambda x: x[1], reverse=True)
                    recent.extend([str(x[0]) for x in items[:20]])
            
            recent_filtered = []
            for fpath in recent:
                try: recent_filtered.append((fpath, float((PROJECT_ROOT / fpath).stat().st_mtime)))
                except: pass
            recent_filtered.sort(key=lambda x: x[1], reverse=True)
            recent = [x[0] for x in recent_filtered[:50]]
            
        except Exception: pass

        cores = [float(c) for c in psutil.cpu_percent(percpu=True)]
        mem = psutil.virtual_memory()
        swp = psutil.swap_memory()
        load = os.getloadavg()
        
        def make_txt_bar(pct, length=15):
            filled = int((pct / 100) * length)
            color = "text-emerald-500" if pct < 60 else "text-amber-500" if pct < 85 else "text-red-500"
            return f'<span class="{color}">{"|" * filled}</span><span class="text-slate-600">{"|" * (length - filled)}</span>'

        htop_data = {
            "cores": cores,
            "mem_str": f"Mem [{make_txt_bar(mem.percent)} {mem.used/(1024**3):.1f}G/{mem.total/(1024**3):.1f}G]",
            "swp_str": f"Swp [{make_txt_bar(swp.percent)} {swp.used/(1024**3):.1f}G/{swp.total/(1024**3):.1f}G]",
            "load": f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}",
            "uptime": format_uptime(time.time() - psutil.boot_time()),
            "tasks": int(len(psutil.pids()))
        }

        payload = {
            "server_uptime": format_uptime(server_uptime),
            "compute_uptime": format_uptime(compute_uptime_sec),
            "is_running": is_running,
            "processes": processes,
            "recent_files": recent,
            "htop": htop_data,
            "sys_history": SYS_HISTORY,
            "discovery_points": get_discovery_points(), 
            "state": state,
            "h5ad_stats": H5AD_STATS,
            "facilities": FACILITY_POINTS
        }
        
        safe_payload = clean_dict(payload)
        return jsonify(safe_payload)
        
    except Exception as e:
        return jsonify({"error": traceback.format_exc()}), 500

@app.route('/stream_logs')
def stream_logs():
    def generate():
        current_log = None
        last_pos = 0
        while True:
            latest = get_latest_log()
            if latest and current_log != latest:
                current_log = latest
                yield f"data: [SYSTEM] Monitoring {current_log.name}\n\n"
                try:
                    with open(current_log, "r", encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in lines[-30:]: yield f"data: {line.strip()}\n\n"
                        last_pos = f.tell()
                except: pass
            try:
                with open(current_log, "r", encoding='utf-8', errors='ignore') as f:
                    f.seek(last_pos)
                    for line in f.read().splitlines():
                        if line.strip(): yield f"data: {line.strip()}\n\n"
                    last_pos = f.tell()
            except: pass
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)