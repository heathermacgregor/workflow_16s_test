Here is a comprehensive `README.md` for the **Nuclear Fuel Cycle (NFC) Module**. You can place this file in `src/workflow_16s/api/nuclear_fuel_cycle/README.md`.

---

# Nuclear Fuel Cycle (NFC) Spatial Annotation Module

## 🌍 Overview

The **NFC Module** is a specialized geospatial pipeline designed to biologically contextualize microbiome samples. It aggregates global industrial facility data (nuclear and non-nuclear) and projects this metadata onto biological samples based on spatial proximity.

**Core Question Answered:** *"Is this sample located near a nuclear facility, an industrial analog (e.g., gold mine), or a control site?"*

## 🏗️ Architecture

The pipeline operates as a funnel, aggregating disparate data sources into a unified geospatial index.

1. **Ingest:** Parallel loaders fetch data from APIs (Mindat, Wikidata) and static datasets (GEM, IAEA).
2. **Normalize:** Standardizes columns (`lat`, `lon`, `country`, `facility_type`) and enforces schema.
3. **Geocode:** Resolves missing coordinates for facilities that only list "City, Country".
4. **Spatial Match:** Uses a **KD-Tree** to find the nearest facility for every biological sample.
5. **Annotate:** Injects metadata (`facility_match`, `distance_km`, `facility_type`) into `adata.obs`.

---

## 📊 Data Sources

The module aggregates data from the following sources:

| Source | Type | Description | Key Modules |
| --- | --- | --- | --- |
| **GEM** | Nuclear | Global Energy Monitor (Nuclear Power Plants). | `_gem.py` |
| **IAEA (NFCIS)** | Nuclear | Integrated Nuclear Fuel Cycle Information Systems (Processing, Enrichment). | `_iaea.py` |
| **Mindat** | Nuclear | Worldwide Uranium Mines (API). | `_mindat.py` |
| **NRC / DNFSB** | Nuclear | US Regulatory data for defense & commercial sites. | `_nrc.py`, `_dnfsb.py` |
| **Wikidata** | **Analog** | "Look-alike" industries (Gold/Copper mines, Desalination, Coal Power). | `_analogs.py` |

---

## ⚙️ Configuration

Configure the module in `config.yaml`. You can toggle specific databases to save time or focus the analysis.

```yaml
nfc_facilities:
  enabled: true
  use_cache: true             # Load from .pkl if available (fast)
  distance_threshold_km: 25.0 # Max distance to consider a "match"
  
  # API Keys (Required for Mindat)
  mindat_api_key: "YOUR_KEY_HERE"
  
  # Database Selection (Leave empty to use all)
  databases:
    - "gem"
    - "nfcis"
    - "mindat"
    - "wikidata" # Analog sites
    - "analogs"  # Alias for wikidata

```

---

## 🚀 Usage

### Automatic Integration

In the main workflow (`workflow_16s`), this module is called automatically during the **Ingestion** or **Analysis** phase.

```python
from workflow_16s.api.nuclear_fuel_cycle.nfc import NuclearFuelCycleHandler

# Initialize
nfc_handler = NuclearFuelCycleHandler(config, output_dir)

# 1. Fetch & Aggregate Data
# Returns a dataframe of ~30,000 global facilities
facilities_df = await nfc_handler.nfc_facilities()

# 2. Match Samples
# Annotates the AnnData object in-place
await nfc_handler.match_samples(workflow.adata)

```

### Output in `adata.obs`

After running, your samples will have these new columns:

* **`facility_match`** *(bool)*: True if sample is within `distance_threshold_km` of any facility.
* **`facility_type`** *(str)*: Specific type (e.g., "Nuclear Power Plant", "Analog - Gold Mine").
* **`facility_name`** *(str)*: Name of the nearest facility.
* **`facility_distance_km`** *(float)*: Exact distance to the facility.
* **`facility_category`** *(str)*: "Nuclear Fuel Cycle" or "Contamination Analog".

---

## 📂 File Structure

```text
api/nuclear_fuel_cycle/
├── nfc.py            # CORE: Orchestrator, Geocoding, and Spatial Matching logic
├── _analogs.py       # LOADER: Wikidata scraper for industrial analogs
├── _gem.py           # LOADER: Global Energy Monitor parser
├── _mindat.py        # LOADER: Mindat API client
├── _iaea.py          # LOADER: NFCIS parser
├── _geocode.py       # UTILITY: Geocoding service wrapper
└── README.md         # This file

```