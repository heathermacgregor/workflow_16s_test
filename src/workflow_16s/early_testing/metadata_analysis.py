import pandas as pd
import re
import numpy as np
import requests
import time
import xml.etree.ElementTree as ET
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import logging
logger = logging.getLogger("workflow_16s")

def _standardize_dates(df):
    """
    Finds date-related columns and converts them to a standard YYYY-MM-DD format.
    """
    logger.info("\nStandardizing date formats to YYYY-MM-DD...")
    # Find columns with 'date' or 'time' in their name
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_cols:
        logger.info("  - No date columns found to standardize.")
        return df
    
    for col in date_cols:
        logger.info(f"  - Processing column: {col}")
        # Convert column to datetime objects, coercing any errors into NaT (Not a Time)
        parsed_dates = pd.to_datetime(df[col], errors='coerce')
        # Format the valid dates into 'YYYY-MM-DD' strings, keeping NaT for failed parses
        df[col] = parsed_dates.dt.strftime('%Y-%m-%d')
    
    logger.info("  - Date standardization complete.")
    return df

def _enrich_location_from_coords(df):
    """
    Uses reverse geocoding to find city, state, country for rows with lat/lon but no location text.
    """
    logger.info("\nAttempting to enrich location data from latitude/longitude...")
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        logger.info("  - Latitude/longitude columns not found. Skipping enrichment.")
        return df

    if 'location' not in df.columns:
        df['location'] = np.nan # Ensure location column exists

    # Initialize geolocator - Nominatim requires a custom user_agent
    try:
        geolocator = Nominatim(user_agent="metadata_analysis_script_v1")
    except Exception as e:
        logger.info(f"  - Could not initialize geolocator. Skipping. Error: {e}")
        return df
        
    rows_to_check = df[df['location'].isnull() & pd.to_numeric(df['latitude'], errors='coerce').notna() & pd.to_numeric(df['longitude'], errors='coerce').notna()]
    
    if rows_to_check.empty:
        logger.info("  - No rows require location enrichment.")
        return df

    logger.info(f"  - Found {len(rows_to_check)} rows to enrich with geocoding...")
    
    for index, row in rows_to_check.iterrows():
        lat, lon = row['latitude'], row['longitude']
        logger.info(f"    - Geocoding coordinates: ({lat}, {lon})...")
        
        try:
            # Add a delay to respect Nominatim's usage policy (max 1 request/sec)
            time.sleep(1.1)
            location_data = geolocator.reverse(f"{lat}, {lon}", exactly_one=True)
            # If location_data is a coroutine (async), await it
            if hasattr(location_data, "__await__"):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                if location_data is not None:
                    location_data = loop.run_until_complete(location_data)

            if location_data is not None:
                # Now safe to access .raw
                if hasattr(location_data, 'raw') and 'address' in location_data.raw: # type: ignore
                    address = location_data.raw['address'] # type: ignore
                    # Extract components, preferring city/town/village etc.
                    city = address.get('city', address.get('town', address.get('village', '')))
                    state = address.get('state', '')
                    country = address.get('country', '')
                    
                    # Build a clean, ordered location string
                    loc_parts = [part for part in [city, state, country] if part]
                    enriched_location = ", ".join(loc_parts)
                    
                    df.loc[index, 'location'] = enriched_location
                    logger.info(f"      - Found location: {enriched_location}")
                else:
                    logger.info("      - No location data found for these coordinates.")
            else:
                logger.info("      - No location data found for these coordinates.")

        except GeocoderTimedOut:
            logger.info("      - Geocoding service timed out. Moving on.")
        except GeocoderServiceError as e:
            logger.info(f"      - Geocoding service error: {e}. Moving on.")
        except Exception as e:
            logger.info(f"      - An unexpected error occurred during geocoding: {e}")

    logger.info("  - Location enrichment complete.")
    return df

def _merge_text_fields(df, source_cols, separator, preferred_order=None, split_delimiters=None):
    """
    Intelligently merges multiple text columns into one based on provided configuration.
    It splits content by common delimiters, gets unique non-empty parts, and rejoins them
    after performing semantic deduplication.
    """
    
    if preferred_order:
        def get_order_key(col):
            for i, pattern in enumerate(preferred_order):
                if re.search(pattern, col, re.IGNORECASE):
                    return i
            return len(preferred_order)
        ordered_source_cols = sorted(source_cols, key=get_order_key)
    else:
        ordered_source_cols = source_cols

    def merge_row(row):
        all_parts = []
        for col in ordered_source_cols:
            cell_value = row[col]
            if pd.notna(cell_value) and str(cell_value).strip():
                if split_delimiters:
                    parts = re.split(split_delimiters, str(cell_value))
                    all_parts.extend(parts)
                else:
                    all_parts.append(str(cell_value))
        
        unique_parts = list(dict.fromkeys([p.strip() for p in all_parts if p and p.strip()]))
        
        final_parts = []
        for part in unique_parts:
            is_substring_of_another = any(part != other_part and part in other_part for other_part in unique_parts)
            if not is_substring_of_another:
                final_parts.append(part)

        if not final_parts:
            return np.nan
        
        return separator.join(final_parts)

    existing_source_cols = [col for col in ordered_source_cols if col in df.columns]
    if not existing_source_cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    return df[existing_source_cols].apply(merge_row, axis=1)

def _convert_envo_codes(df):
    """
    Finds columns with ENVO or EMPO codes and converts them to human-readable labels using the OLS API.
    Handles multiple, varied formats in a single cell (e.g., 'ENVO_01000187|ENVO:01000222').
    """
    logger.info("\nConverting ENVO/EMPO codes to labels...")
    
    envo_cols_to_check = ['env_material', 'env_feature', 'env_biome']
    envo_cache = {}  # Shared cache for efficiency
    base_url = "https://www.ebi.ac.uk/ols/api/ontologies/envo/terms"

    def lookup_code(code):
        if code in envo_cache:
            return envo_cache[code]
        
        logger.info(f"    - Looking up new code: {code}...")
        try:
            iri = f"http://purl.obolibrary.org/obo/{code.replace(':', '_')}"
            params = {'iri': iri}
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(0.4)
            
            data = response.json()
            terms = data.get('_embedded', {}).get('terms', [])
            if terms and 'label' in terms[0]:
                label = terms[0]['label']
                envo_cache[code] = label
                logger.info(f"      - Found label: '{label}'")
            else:
                envo_cache[code] = code
        except Exception as e:
            logger.info(f"      - API request failed for {code}: {e}")
            envo_cache[code] = code
        return envo_cache[code]

    def parse_and_convert_cell(cell_content):
        if pd.isna(cell_content) or not isinstance(cell_content, str):
            return cell_content

        parts = re.split(r'[|;,]\s*', cell_content)
        converted_parts = []
        
        for part in parts:
            part = part.strip()
            match = re.search(r'(\d{7,})', part)
            if match:
                code_num = match.group(1)
                standard_code = f"ENVO:{code_num}"
                label = lookup_code(standard_code)
                converted_parts.append(label)
            else:
                converted_parts.append(part)
        
        return "; ".join(list(dict.fromkeys(p for p in converted_parts if p)))

    for envo_col in envo_cols_to_check:
        if envo_col in df.columns:
            logger.info(f"\n  - Processing column '{envo_col}' for ENVO codes...")
            df[envo_col] = df[envo_col].apply(parse_and_convert_cell)

    logger.info("\nENVO conversion process complete.")
    return df

def _find_publications(df, api_key=None):
    """
    Attempts to find publication DOIs for samples missing them, using accession numbers.
    It uses a two-stage search and an internal cache to avoid redundant API calls.
    """
    logger.info("\nAttempting to find missing publications via NCBI E-utils...")

    if 'publication_doi' not in df.columns:
        df['publication_doi'] = np.nan

    accession_pattern = r'^(run|sample|experiment|study|project|sra|ena|ddbj|biosample|bioproject)_?(accession|alias)$|^accession$'
    all_accession_cols = [col for col in df.columns if re.search(accession_pattern, col, re.IGNORECASE)]

    if not all_accession_cols:
        logger.info("  - No accession columns found, cannot search for publications.")
        return df
    
    # Prioritize which accession column to use for the search
    priority_order = ['project', 'study', 'biosample', 'sra', 'ena', 'sample', 'experiment', 'run']
    sorted_accession_cols = sorted(
        all_accession_cols, 
        key=lambda col: next((i for i, p in enumerate(priority_order) if p in col.lower()), len(priority_order))
    )
    
    rows_to_search = df[df['publication_doi'].isnull()]
    if rows_to_search.empty:
        logger.info("  - No samples require publication search.")
        return df

    logger.info(f"  - Found {len(rows_to_search)} samples to check for publications.")
    
    # --- Caching Implementation Start ---
    # This dictionary will store results to avoid re-querying the same accession
    accession_cache = {} 
    # --- Caching Implementation End ---
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    api_key_str = f"&api_key={api_key}" if api_key else ""
    delay = 0.15 if api_key else 0.4 # Shorter delay with API key, longer without

    if not api_key:
        logger.info("  - WARNING: No NCBI API key provided. Proceeding with lower rate limits.")
    
    for index, row in rows_to_search.iterrows():
        # Find the best available accession for this row
        accession = next((row[col] for col in sorted_accession_cols if col in row and pd.notna(row[col])), None)
        
        if not accession:
            continue

        # --- Caching Logic: Check cache before making an API call ---
        if accession in accession_cache:
            cached_doi = accession_cache[accession]
            if cached_doi:
                df.loc[index, 'publication_doi'] = cached_doi
            continue # Move to the next row
        # --- End Caching Logic ---

        logger.info(f"    - Searching for publication linked to new accession: {accession}...")
        found_doi = None
        
        try:
            # Stage 1: Search for formal links in NCBI databases
            uid, db_found_in = None, None
            for db in ['bioproject', 'sra', 'biosample']:
                search_url = f"{base_url}esearch.fcgi?db={db}&term={accession}&retmode=xml{api_key_str}"
                response = requests.get(search_url)
                response.raise_for_status()
                time.sleep(delay)
                root = ET.fromstring(response.content)
                id_elem = root.find('.//Id')
                if id_elem is not None and id_elem.text:
                    uid, db_found_in = id_elem.text, db
                    break
            
            pmids = []
            if uid:
                link_url = f"{base_url}elink.fcgi?dbfrom={db_found_in}&db=pubmed&id={uid}&retmode=xml{api_key_str}"
                response = requests.get(link_url)
                response.raise_for_status()
                time.sleep(delay)
                root = ET.fromstring(response.content)
                pmids = [id_elem.text for id_elem in root.findall(".//LinkSetDb[DbTo='pubmed']//Id") if id_elem.text]
            
            # Stage 2: If no formal links, search PubMed directly for the accession
            if not pmids:
                pubmed_search_url = f"{base_url}esearch.fcgi?db=pubmed&term={accession}&retmode=xml{api_key_str}"
                response = requests.get(pubmed_search_url)
                response.raise_for_status()
                time.sleep(delay)
                root = ET.fromstring(response.content)
                pmids = [id_elem.text for id_elem in root.findall('.//Id') if id_elem.text]

            # If a publication was found, get its DOI
            if pmids:
                pmid = pmids[0] # Use the first result
                summary_url = f"{base_url}esummary.fcgi?db=pubmed&id={pmid}&retmode=xml{api_key_str}"
                response = requests.get(summary_url)
                response.raise_for_status()
                time.sleep(delay)
                root = ET.fromstring(response.content)
                doi_element = root.find(".//Item[@Name='DOI']")
                if doi_element is not None and doi_element.text:
                    found_doi = doi_element.text
                    df.loc[index, 'publication_doi'] = found_doi
                    logger.info(f"      - SUCCESS: Found DOI: {found_doi}")

        except requests.exceptions.RequestException as e:
            logger.info(f"      - A network error occurred for {accession}: {e}. Halting search.")
            break # Stop searching if the network fails
        except ET.ParseError as e:
            logger.info(f"      - Failed to parse XML response for {accession}: {e}")
        
        accession_cache[accession] = found_doi
        if not found_doi:
             logger.info(f"      - No publication found for {accession}.")
            
    return df


def _extract_from_unmapped_cells(df, processed_df, unmapped_cols):
    logger.info("\nAttempting to extract data from unmapped cell content...")
    
    EXTRACTION_MAP = {
        'ph': (r'(?i)(?:ph|p H)[\s:]*(\d{1,2}\.?\d*)', 'ph'),
        'temperature': (r'(?i)temp(?:erature)?[\s:]*(-?\d+\.?\d*)', 'temperature_conc'),
        'salinity': (r'(?i)salinity[\s:]*(\d+\.?\d*)\s*(?:psu)?', 'salinity_conc'),
        'nitrate': (r'(?i)nitrate|no3[\s:]*(\d+\.?\d*)', 'nitrate_conc'),
        'phosphate': (r'(?i)phosphate|po4[\s:]*(\d+\.?\d*)', 'phosphate_conc'),
    }

    # Ensure the index is set correctly on the original dataframe
    if '#sampleid' not in df.columns:
        logger.info("  - Original dataframe is missing '#sampleid' column. Cannot perform extraction.")
        return processed_df
    original_df_indexed = df.set_index('#sampleid')
    
    for index, row in processed_df.iterrows():
        sample_id = row['#sampleid']
        
        # --- FIX: Skip any row that has a missing or null sample ID ---
        if pd.isna(sample_id):
            continue
        # --- END FIX ---
        
        if sample_id not in original_df_indexed.index: 
            continue
        
        for base_name, (pattern, target_col) in EXTRACTION_MAP.items():
            if target_col in processed_df.columns and pd.isna(row[target_col]):
                for um_col in unmapped_cols:
                    if um_col in original_df_indexed.columns:
                        cell_content = str(original_df_indexed.loc[sample_id, um_col])
                        match = re.search(pattern, cell_content)
                        if match:
                            extracted_value = match.group(1)
                            processed_df.loc[index, target_col] = pd.to_numeric(extracted_value, errors='coerce')
                            logger.info(f"  - Found '{base_name}' value '{extracted_value}' for sample '{sample_id}' in column '{um_col}'")
                            break # Move to the next extraction type once a value is found
                            
    return processed_df

def _parse_alpha_diversity(df, unmapped_cols):
    logger.info("\nParsing alpha diversity columns...")
    alpha_cols_to_drop = [col for col in unmapped_cols if col.startswith('alpha_')]
    if not alpha_cols_to_drop:
        logger.info("  - No alpha diversity columns found to parse.")
        return df, unmapped_cols

    df_copy = df.copy()
    for col in alpha_cols_to_drop:
        match = re.search(r'alpha_(.+)_(shannon|richness|faithspd|fishersalpha)', col)
        if match:
            target, metric = match.groups()
            new_col_name = f"alpha_{metric}_{target}"
            if new_col_name not in df_copy.columns:
                 df_copy[new_col_name] = df_copy[col]
            else:
                 df_copy[new_col_name].fillna(df_copy[col], inplace=True)
            logger.info(f"  - Parsed '{col}' -> '{new_col_name}'")

    df_copy.drop(columns=alpha_cols_to_drop, inplace=True, errors='ignore')
    unmapped_cols = [c for c in unmapped_cols if not c.startswith('alpha_')]
    return df_copy, unmapped_cols

def _consolidate_units(df, base_name, unit_pattern_map):
    logger.info(f"\nConsolidating units for '{base_name}'...")
    
    stat_patterns = {'_std_dev': r'_std_dev|_stddev|_sd$', '_avg': r'_avg|_average$'}
    value_col, unit_col = f"{base_name}_conc", f"{base_name}_conc_unit"
    if value_col not in df.columns: df[value_col] = np.nan
    if unit_col not in df.columns: df[unit_col] = ''

    found_any, cols_to_drop = False, []
    columns_to_check = list(df.columns)

    for col in columns_to_check:
        for pattern, unit in unit_pattern_map.items():
            if re.fullmatch(pattern, col, re.IGNORECASE):
                found_any = True
                stat_suffix_found = next((suffix for suffix, p in stat_patterns.items() if re.search(p, col, re.IGNORECASE)), None)
                numeric_values = pd.to_numeric(df[col], errors='coerce')

                if stat_suffix_found:
                    stat_col_name = f"{base_name}_conc{stat_suffix_found}"
                    if stat_col_name not in df.columns: df[stat_col_name] = np.nan
                    logger.info(f"  - Found stat data in '{col}', consolidating to '{stat_col_name}'")
                    df[stat_col_name].fillna(numeric_values, inplace=True)
                else:
                    logger.info(f"  - Found data in '{col}', assigning unit '{unit}'")
                    update_mask = df[value_col].isnull() & numeric_values.notna()
                    df.loc[update_mask, value_col] = numeric_values[update_mask]
                    df.loc[update_mask, unit_col] = unit
                
                cols_to_drop.append(col)
                break
    
    df.drop(columns=list(set(cols_to_drop)), inplace=True, errors='ignore')

    if not found_any:
        logger.info(f"  - No columns found for '{base_name}'.")
        df.drop(columns=[value_col, unit_col], inplace=True, errors='ignore')
    
    for suffix in stat_patterns:
        stat_col_name = f"{base_name}_conc{suffix}"
        if stat_col_name in df.columns and df[stat_col_name].isnull().all():
            df.drop(columns=[stat_col_name], inplace=True)
            
    return df, list(set(cols_to_drop))

def analyze_metadata(input_filepath, output_filepath, ncbi_api_key=None):
    """
    Intelligently analyzes a TSV metadata file to extract and clean key information.
    """
    logger.info(f"Reading metadata file from: {input_filepath}")
    try:
        df = pd.read_csv(input_filepath, sep='\t', dtype=str, low_memory=False)
    except FileNotFoundError:
        logger.info(f"Error: The file '{input_filepath}' was not found."); return
    except Exception as e:
        logger.info(f"An error occurred while reading the file: {e}"); return

    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}.{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols

    # Define a more robust regex for common sample ID column names
    primary_pattern = re.compile(r'^#?sample([-_]?(id|name))?$', re.I)

    # First, try to find a column matching common sample ID patterns
    sample_id_col = next((col for col in df.columns if primary_pattern.match(col)), None)

    # If no primary match is found, look for a column containing 'run_accession'
    if not sample_id_col:
        sample_id_col = next((col for col in df.columns if 'run_accession' in col.lower()), None)

    # If still no column is found, log an error and return
    if not sample_id_col:
        logger.error("Error: Could not find a suitable sample ID column.")
        return
    
    df.rename(columns={sample_id_col: '#sampleid'}, inplace=True)

    KEYWORD_MAP = {
        'location': r'^(geo_loc_name|site_name|location|country|state|state_or_province|city|sample_site)$',
        'latitude': r'^(lat|latitude|latitude_deg)$',
        'longitude': r'^(lon|longitude|longitude_deg)$',
        'depth_m': r'^(depth|depth_m|water_depth|bottom_depth_cm)$',
        'elevation_m': r'^(altitude|elevation|altitude_m|elevation_m)$',
        'collection_date': r'^(collection_date|collection_timestamp|sampling_date)$',
        'accession': r'^(run|sample|experiment|study|project|sra|ena|ddbj|biosample|bioproject)_?(accession|alias)$|^accession$',
        'publication_doi': r'^(doi|publication_doi|publication_url)$',
        'sequencing_platform': r'^(instrument_model|instrument_platform|sequencing_platform)$',
        'library_strategy': r'^(library_layout|library_strategy)$',
        'extraction_method': r'^(extraction_kit|extraction_method|dna_extraction_kit)$',
        'host_name': r'^(host|host_scientific_name|host_common_name)$',
        'soil_type': r'^(soil_type|soil_class|fao_class|soil_texture)$',
        'env_material': r'^(envo|empo_3|environment_material)$',
        'env_feature': r'^environment_feature$',
        'env_biome': r'^environment_biome$',
        'ph': r'^ph$',
        'project_name': r'^project_name$',
        'principal_investigator': r'^principal_investigator$',
        'alternative_id': r'^alt_id_\d$',
        'host_sex': r'^host_sex$',
        'host_age': r'^host_age$',
        'treatment': r'^treatment$',
        'collected_by': r'^collected_by$',
        'sequencing_method': r'^sequencing_method$',
        'description': r'description|notes|biological_sample_notes',
        'nuclear_contamination_status': r'^nuclear_contamination_status$',
        'nuclear_contamination_level': r'^nuclear_contamination_level$',
        'nuclear_contamination_source': r'^nuclear_contamination_source$',
        'facility_match': r'^facility_match$',
        'facility_distance_km': r'^facility_distance_km$',
        'facility_description': r'^facility_description$',
        'target_gene': r'^target_gene$',
        'target_subfragment': r'^target_subfragment$',
        'target_subfragment_length': r'^target_subfragment_length$'
    }
    
    MERGE_CONFIG = {
        'location': { 'separator': ', ', 'preferred_order': [r'site', r'city', r'state', r'country', r'loc'], 'split_delimiters': r'[,:]\s*' },
        'sequencing_platform': { 'separator': ' ', 'split_delimiters': None },
        'description': { 'separator': '. ', 'split_delimiters': r'[.;]\s*' },
        'alternative_id': { 'separator': '; ', 'split_delimiters': None }
    }

    column_mapping, unmapped_cols = {}, []
    logger.info("\nAnalyzing columns by header...")
    for col_name in df.columns:
        if col_name == '#sampleid': continue
        found_map = False
        for sensible_name, pattern in KEYWORD_MAP.items():
            if re.search(pattern, col_name, re.IGNORECASE):
                column_mapping.setdefault(sensible_name, []).append(col_name)
                found_map = True
        if not found_map: unmapped_cols.append(col_name)
    
    clean_df = df[['#sampleid']].copy()
    for sensible_name, original_cols in column_mapping.items():
        if sensible_name in MERGE_CONFIG:
            logger.info(f"  - Intelligently merging '{sensible_name}' <-- {original_cols}")
            clean_df[sensible_name] = _merge_text_fields(df, original_cols, **MERGE_CONFIG[sensible_name])
        elif 'accession' in sensible_name:
            logger.info(f"  - Preserving individual accession columns: {original_cols}")
            for col in original_cols:
                if col not in clean_df.columns: clean_df[col] = df[col]
        else:
            logger.info(f"  - Coalescing '{sensible_name}' <-- {original_cols}")
            clean_df[sensible_name] = df[original_cols].bfill(axis=1).iloc[:, 0]

    clean_df = _standardize_dates(clean_df)
    
    if 'nuclear_contamination_status' in clean_df.columns:
        logger.info("  - Standardizing 'nuclear_contamination_status' column to boolean...")
        true_values = ['true', 'yes', 'contaminated', '1']
        clean_df['nuclear_contamination_status'] = clean_df['nuclear_contamination_status'].astype(str).str.lower().isin(true_values)
    
    clean_df = _enrich_location_from_coords(clean_df)
    clean_df = _convert_envo_codes(clean_df)

    temp_df = pd.concat([clean_df.set_index('#sampleid'), df[unmapped_cols].set_index(df['#sampleid'])], axis=1).reset_index()
    temp_df, unmapped_cols = _parse_alpha_diversity(temp_df, unmapped_cols)

    CONSOLIDATION_MAP = {
        'temperature': {r'temp_celsius|temperature|air_temp|annual_season_temp|soil_temp_ave': 'celsius'},
        'salinity': {r'salinity|sal_psu|soil_salinity_ave': 'psu'},
    }

    processed_df = temp_df
    all_dropped_cols = []
    for base_name, unit_pattern_map in CONSOLIDATION_MAP.items():
        processed_df, dropped_cols = _consolidate_units(processed_df, base_name, unit_pattern_map)
        all_dropped_cols.extend(dropped_cols)
    
    processed_df = _find_publications(processed_df, api_key=ncbi_api_key)

    unmapped_cols = [c for c in unmapped_cols if c not in all_dropped_cols and '.1' not in c]
    
    processed_df = _extract_from_unmapped_cells(df, processed_df, unmapped_cols)
    final_unmapped = [c for c in unmapped_cols if c in processed_df.columns]
    drop_unmapped = False
    if drop_unmapped and final_unmapped:
        logger.info(f"\nDropping unmapped columns: {final_unmapped}")
        processed_df.drop(columns=final_unmapped, inplace=True, errors='ignore')
        logger.info(f"\nRemaining unmapped columns that were dropped: {final_unmapped if final_unmapped else 'None'}")
    logger.info(f"\nWriting cleaned data to: {output_filepath}")
    processed_df.to_csv(output_filepath, sep='\t', index=False)
    logger.info("Analysis complete.")


if __name__ == '__main__':
    import sys
    # --- IMPORTANT ---
    # For the publication search to work reliably, get a free API key from your NCBI account:
    # 1. Go to https://www.ncbi.nlm.nih.gov/
    # 2. Log in to your account (or create one).
    # 3. Go to Account Settings -> API Key Management -> Create an API Key.
    # 4. Paste the key below.
    YOUR_NCBI_API_KEY = "4c94f710e470be18d4ef4de1debb0d804c09" 
    analyze_metadata('/usr2/people/macgregor/amplicon/test/data/merged/metadata/final_metadata.tsv', '/usr2/people/macgregor/amplicon/test/data/merged/metadata/cleaned_metadata.tsv')
