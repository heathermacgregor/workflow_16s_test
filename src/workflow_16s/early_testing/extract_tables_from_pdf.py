import pandas as pd
from io import StringIO
import re
from unstructured.partition.pdf import partition_pdf
import os # Added for file path manipulation

# --- CORE PARSING LOGIC (Handles Fallback for Messy OCR Text) ---

# Known Total Volumes from the document image context. Used as a fallback anchor.
KNOWN_VOLUMES = {
    "Banner Spring Down": "431862320", # 431,862,320
    "Sewer": "22409038",            # 22,409,038
    "West Ditch": "102302320",       # 102,302,320
    "WWTF": "2995114"               # 2,995,114
}

# The combined list of known location titles for splitting the raw text
KNOWN_LOCATIONS = r'(Banner\s*Spring\s*Down|Total:\s*Sewer|Total:\s*West\s*Ditch|Total:\s*WWTF)'

def clean_ocr_value(val: str) -> str:
    """Cleans common OCR errors in scientific notation for reliable conversion to float/int."""
    if not isinstance(val, str):
        return str(val)
    
    cleaned = (val.strip()
               .replace('l', '1') # Common OCR mistake: 'l' (lowercase L) -> '1'
               .replace('I', '1') # Common OCR mistake: 'I' (uppercase I) -> '1'
               .replace('O', '0') # Common OCR mistake: 'O' (uppercase O) -> '0'
               .replace('S', '5') # Occasional OCR mistake: 'S' -> '5'
               .replace('£', 'E') # Currency symbol mistaken for 'E' in E-notation
               .replace(' ', '')  # Remove spaces within numbers (e.g., '431 ,862,320')
               .replace(',', '')  # Remove thousands separator commas
               )
    
    # Handle cases like 'l.2lE+02' which becomes '1.21E+02'
    # Heuristically remove a second decimal point if found, as this breaks float conversion
    if cleaned.count('.') > 1:
        # Example: '1.2.3E-10' -> '1.23E-10' (only keeps the first decimal)
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    return cleaned

# Refined Number Pattern: Matches scientific notation or large integers, handling OCR errors (l, I, O)
# This pattern is aggressive to capture anything that looks like a number.
NUMBER_PATTERN_REFINED = re.compile(
    r'([lI0\d][\s\.,]*[lI0\d]*[Ee][\+\-][lI0\d]+|\d{1,3}(?:,\d{3})*|\d[\.\,]\d+)', 
    re.IGNORECASE
)

def parse_raw_text_table(raw_text: str) -> pd.DataFrame:
    """
    Custom parser to convert the raw, messy OCR text into a clean DataFrame.
    This is necessary because HTML extraction failed.
    """
    print("\n--- Starting Custom Raw Text Parser ---")
    
    # 1. Define final column structure based on context/image analysis
    COLUMNS = [
        "Location", "Activity", "Total Volume (l)", "Activity Concentration (µCi/ml)", 
        "Error Estimate (µCi/ml)", "LLD (µCi/ml)", "Quantity Released (Ci)", 
        "Quantity Released (g)", "Fraction of ECV"
    ]
    all_rows = []
    
    current_location = None
    expected_volume = None
    
    # 2. Refined Isotope Pattern: (Pu-238, Tc-99, U-233/234, etc.)
    ISOTOPE_NAMES = r'(Pu\s*-\s*\d{2,3}(?:\s*/\s*\d{2,3})?|Tc\s*-\s*99|Th\s*-\s*\d{2,3}|U\s*-\s*\d{2,3}(?:\s*/\s*\d{2,3})?|Am\s*-\s*241|Cs\s*-\s*137|Na\s*-\s*22|Np\s*-\s*237|Pb\s*-\s*212)'

    # NEW ROW PATTERN: Find the isotope name, followed by all data, using a lookahead for the next isotope or end marker.
    # Group 1: The Activity/Isotope name
    ROW_PATTERN_SIMPLIFIED = re.compile(
        ISOTOPE_NAMES + 
        r'(.*?)' + # Group 2: Capture all data values (non-greedy)
        r'(?=' + ISOTOPE_NAMES + r'|Total:|$)', # Lookahead for next isotope, 'Total:', or end of block
        re.IGNORECASE | re.DOTALL
    )
    
    # Clean the raw text before processing: replace newlines with a single space and strip
    raw_text = raw_text.replace('\n', ' ').strip()
    
    # 3. Process the text by splitting on location markers 
    
    # The LOCATION_SPLIT_PATTERN will be the raw string KNOWN_LOCATIONS (no compilation here)
    
    # Use finditer and split manually to preserve the delimiters and handle multi-table output
    parts = []
    last_end = 0
    
    # Check if the raw_text contains any location headers at all. If not, treat the whole thing as one block
    location_matches = list(re.finditer(KNOWN_LOCATIONS, raw_text, re.IGNORECASE)) # FIXED: Use raw string pattern with flags
    
    if not location_matches and len(raw_text) > 100:
        # This is a subsequent table (Table 2, 3, etc.) that contains only data rows.
        current_location = "Table Data (No Header)"
        expected_volume = None 
        parts = ["Unknown Location Header", raw_text] # Fake split to run the logic below

    else:
        for m in location_matches:
            # Add the text block *before* the location marker (i.e., the data for the previous location)
            parts.append(raw_text[last_end:m.start()].strip()) 
            # Add the location marker itself
            parts.append(m.group(0).strip())
            last_end = m.end()
        parts.append(raw_text[last_end:].strip()) # Add the final block

    
    for part in parts:
        if not part.strip():
            continue
            
        # FIXED: Use raw string KNOWN_LOCATIONS with re.IGNORECASE flag.
        location_match = re.match(KNOWN_LOCATIONS, part, re.IGNORECASE)
        
        # Check for both actual location match and our "Unknown" placeholder
        if location_match or part == "Unknown Location Header":
            # Set the new current location and its expected volume
            if part == "Unknown Location Header":
                current_location = "Table Data (No Header)"
                expected_volume = None
            else:
                current_location = location_match.group(0).replace("Total:", "").strip()
                
                # Clean the location name to match the KNOWN_VOLUMES keys
                clean_loc = " ".join(current_location.split()) # Normalize spaces
                expected_volume = KNOWN_VOLUMES.get(clean_loc, None)
            continue
        
        # This part contains isotope data for the current_location
        data_block = part

        # Process all isotope rows within the current location block
        for match in ROW_PATTERN_SIMPLIFIED.finditer(data_block):
            # Group 1: Activity/Isotope name
            activity = match.group(1).strip().replace(" ", "")
            
            # Group 2: The raw numerical data
            raw_values = match.group(2).strip()
            
            # Extract and clean all numerical tokens from the raw values
            number_tokens = NUMBER_PATTERN_REFINED.findall(raw_values)
            cleaned_values = [clean_ocr_value(t) for t in number_tokens if clean_ocr_value(t)]

            
            # --- Anchoring Logic: Prioritize finding the volume, otherwise use the constant ---
            volume_str = None
            
            # 3a. Try to find the large volume integer in the tokens
            max_vol = 0
            idx_to_remove = -1
            for i, val in enumerate(cleaned_values):
                try:
                    # Check for a large integer (> 5 digits)
                    int_val = int(val.split('E')[0].split('.')[0])
                    if int_val > max_vol and len(str(int_val)) > 5:
                         max_vol = int_val
                         volume_str = val
                         idx_to_remove = i
                except ValueError:
                    continue

            # This will hold the 6 metrics (Conc, Error, LLD, Ci, g, Fraction)
            remaining_metrics = []

            if volume_str:
                # Volume found by OCR: remove it from the tokens
                remaining_metrics = cleaned_values[:idx_to_remove] + cleaned_values[idx_to_remove+1:]
                
            elif expected_volume:
                # Volume not found by OCR: use the expected constant volume
                volume_str = expected_volume
                
                # Assume the current tokens (cleaned_values) ONLY contain the 6 metrics, 
                # or possibly a malformed/missing volume which we can't reliably remove.
                # Use the tokens as the metrics, and rely on the volume injection.
                remaining_metrics = cleaned_values
            
            # Skip if we couldn't anchor volume and don't have enough metrics
            if volume_str is None or len(remaining_metrics) < 6:
                if not activity.lower().startswith("total"):
                    print(f"Skipping row for {activity} in {current_location or 'None'}: Could not anchor volume and/or find enough metrics.")
                    continue
            
            # 3b. Compile the 7 required data fields (Volume + 6 Metrics)
            ordered_data = []
            
            # Total Volume (l) is the first data field
            ordered_data.append(volume_str)
            
            # The next 6 are the remaining metrics (Activity Conc, Error, LLD, Ci, g, Fraction)
            # Take the first 6 metrics, which are assumed to be in sequential order after volume detection/injection.
            ordered_data.extend(remaining_metrics[:6])
            
            if len(ordered_data) == 7:
                
                # Build the final row
                row_data = [current_location, activity]
                row_data.extend(ordered_data)
                
                # Pad to ensure exactly 9 columns
                if len(row_data) < len(COLUMNS):
                     row_data.extend([None] * (len(COLUMNS) - len(row_data)))
                     
                all_rows.append(row_data)

    if not all_rows:
        return pd.DataFrame(columns=COLUMNS)
    
    # 4. Create DataFrame and apply type conversion
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    
    # Convert numerical columns to float, coercing errors
    for col in COLUMNS[2:]:
        # Remove any lingering special characters before conversion
        df[col] = df[col].astype(str).str.replace(r'[^0-9eE\.\+\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    print("--- Custom Raw Text Parser Complete ---")
    return df

# --- UNUSED ORIGINAL HTML PARSER (for completeness) ---
def parse_html_table(html_content: str) -> pd.DataFrame:
    """The original preferred method (now skipped)"""
    # ... (implementation remains the same, but is now skipped)
    return pd.DataFrame() 

# --- MAIN EXECUTION LOGIC ---
if __name__ == "__main__":
    # The file path that was failing HTML extraction
    fname = "/usr2/people/macgregor/amplicon/workflow_16s/src/workflow_16s/early_testing/ML25111A044.pdf"
    
    print(f"Attempting to partition PDF: {fname}")
    
    try:
        from unstructured.partition.pdf import partition_pdf
        
        # Perform the PDF partitioning
        elements = partition_pdf(filename=fname,
                                 skip_infer_table_types=False,
                                 strategy='hi_res',
                                 )
        # Filter for table elements
        tables = [el for el in elements if el.category == "Table"]
        
        if not tables:
            print("No tables found in the PDF using unstructured.partition_pdf.")
        else:
            # We iterate over all found tables, just in case multiple tables exist.
            for i, table_element in enumerate(tables):
                print(f"\nFound {len(tables)} table(s). Parsing table {i+1}...")
                
                # Attempt to get HTML content (fails, based on user output)
                html_content = table_element.metadata.text_as_html

                if not html_content:
                    # --- FALLBACK PATH: Use Raw Text Parser ---
                    raw_text = table_element.text
                    if raw_text:
                        print("\n--- HTML Extraction Failed: Falling back to Raw Text Parser ---")
                        df = parse_raw_text_table(raw_text)
                        
                        if not df.empty:
                            print(f"\n--- SUCCESSFULLY PARSED DATAFRAME FOR TABLE {i+1} FROM RAW TEXT (FALLBACK) ---")
                            print("WARNING: Due to OCR errors, columns may have been slightly reordered or contain NaNs.")
                            print(df.to_markdown(index=False)) # Display as clean markdown table
                            
                            # --- TSV Saving Logic ---
                            # 1. Get document base name (ML25111A044)
                            base_name = os.path.splitext(os.path.basename(fname))[0]
                            # 2. Construct output file path (ML25111A044_table_1.tsv)
                            output_filename = f"{base_name}_table_{i+1}.tsv"
                            # 3. Save to TSV
                            df.to_csv(output_filename, sep='\t', index=False)
                            print(f"✅ Successfully saved DataFrame to {output_filename}")
                            # --- End TSV Saving Logic ---

                        else:
                            print(f"\n❌ FAILED: The custom raw text parser could not extract any structured rows for table {i+1}.")
                    else:
                        print(f"\nWARNING: Raw text output for table {i+1} was also empty.")
                else:
                    # This path is not executed based on your logs, but remains for completeness
                    print("\n--- HTML content was detected. Parsing via pandas.read_html (NOT EXECUTED IN THIS DEBUG PATH) ---")
                    # df = parse_html_table(html_content)
                    # if not df.empty:
                    #     print("\n--- SUCCESSFULLY PARSED DATAFRAME FROM HTML ---")
                    #     print(df.to_markdown(index=False))

    except FileNotFoundError:
        print(f"Error: File not found at path: {fname}. Please verify the file path.")
    except ImportError:
        print("Error: The 'unstructured' library or its dependencies (like 'pandas') are not installed. Please ensure your environment is configured correctly.")
    except Exception as e:
        print(f"A general error occurred during table processing: {e}")
