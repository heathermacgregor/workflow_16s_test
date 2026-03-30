# workflow_16s/api/publication/extractors/llm_analyzer.py

import json
import re
import requests
from typing import Dict, Any, List

class MethodologyAnalyzer:
    def __init__(self, api_key: str, logger):
        self.api_key = api_key
        self.logger = logger
        self.endpoint = "https://models.inference.ai.azure.com/chat/completions"
        self.model = "meta-llama-3.1-70b-instruct"

    def _verify_against_source(self, extracted_items: List[str], source_text: str, is_dna: bool = False) -> List[str]:
        """
        Anti-hallucination shield: Verifies that extracted items actually exist in the source text.
        """
        if not extracted_items or not source_text:
            return []

        verified_items = []
        
        # Normalize the source text for standard text comparison
        norm_source = re.sub(r'\s+', ' ', source_text).lower()
        
        # Strip all formatting from source text for pure DNA sequence matching
        if is_dna:
            dna_source = re.sub(r'[^a-zA-Z]', '', source_text).upper()

        for item in extracted_items:
            # 1. Verification for DNA Sequences (Primers)
            if is_dna:
                clean_seq = re.sub(r'[^a-zA-Z]', '', item).upper()
                # Must be at least 10 bases to be considered a valid primer sequence to check
                if len(clean_seq) >= 10 and clean_seq in dna_source:
                    verified_items.append(clean_seq)
                else:
                    self.logger.warning(f"Hallucination caught! Dropped primer '{item}' (Not found in source).")
            
            # 2. Verification for Text (Kits, Regions, Models)
            else:
                norm_item = re.sub(r'\s+', ' ', str(item)).lower()
                # We require the extracted item (or a highly similar subset of it) to be in the text
                if norm_item in norm_source:
                    verified_items.append(item)
                else:
                    # Try a fuzzy fallback: Check if the longest word of the item is in the text
                    words = [w for w in norm_item.split() if len(w) > 4]
                    if words and any(w in norm_source for w in words):
                        verified_items.append(item) # Partial match accepted
                    else:
                        self.logger.warning(f"Hallucination caught! Dropped text '{item}' (Not found in source).")

        return verified_items
    
    def _extract_methodology_details_llm(self, text_to_scan: str) -> Dict[str, Any]:
        """
        Extracts comprehensive methodology details using an LLM.
        Strictly enforces JSON schema and runs an anti-hallucination verification pass.
        """
        if not self.api_key:
            return self._extract_methodology_details(text_to_scan)
            
        text_chunk = text_to_scan[:20000] 
        
        # 1. UPGRADED PROMPT: Explicitly forbid guessing
        system_prompt = (
            "You are an expert bioinformatician data extractor. Read the provided materials and methods text "
            "and extract the experimental details. Return ONLY a raw JSON object with absolutely no markdown formatting. "
            "CRITICAL: DO NOT GUESS OR INFER. EXTRACT EXACT STRINGS FROM THE TEXT. IF IT IS NOT EXPLICITLY WRITTEN, RETURN AN EMPTY LIST.\n"
            "The JSON must contain exactly these keys:\n"
            "- 'sample_storage' (list of strings)\n"
            "- 'extraction_protocol_and_kits' (list of strings)\n"
            "- 'pcr_conditions_and_kits' (list of strings)\n"
            "- 'primer_names' (list of strings)\n"
            "- 'primer_sequences' (list of strings)\n"
            "- 'variable_regions' (list of strings)\n"
            "- 'sequencing_details' (list of strings)\n"
            "- 'unextracted_flag' (boolean): Set to true ONLY if the text explicitly references methodology "
            "(e.g., 'primers are listed in Table S1') that is MISSING from this text.\n"
            "- 'unextracted_reason' (string): If unextracted_flag is true, briefly explain what is referenced. Else, empty string."
        )
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama-3.1-70b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze:\n{text_chunk}"}
            ],
            "temperature": 0.0 # Force maximum determinism
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers, 
                json=payload, 
                timeout=45
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            content = content.replace("```json", "").replace("```", "").strip()
            llm_results = json.loads(content)
            
            # Define which keys get which validation treatment
            text_keys = [
                'sample_storage', 'extraction_protocol_and_kits', 'pcr_conditions_and_kits', 
                'primer_names', 'variable_regions', 'sequencing_details'
            ]
            for k in text_keys:
                if k in llm_results and isinstance(llm_results[k], list):
                    # Verify text items
                    llm_results[k] = self._verify_against_source(llm_results[k], text_chunk, is_dna=False)
                else:
                    llm_results[k] = []
                    
            if 'primer_sequences' in llm_results and isinstance(llm_results['primer_sequences'], list):
                # Verify DNA sequences strictly
                llm_results['primer_sequences'] = self._verify_against_source(llm_results['primer_sequences'], text_chunk, is_dna=True)
            else:
                llm_results['primer_sequences'] = []
                    
            if 'unextracted_flag' not in llm_results:
                llm_results['unextracted_flag'] = False
            if 'unextracted_reason' not in llm_results:
                llm_results['unextracted_reason'] = ""
                    
            return llm_results
            
        except Exception as e:
            self.logger.debug(f"LLM extraction failed ({type(e).__name__}: {e}). Falling back to regex.")
            return self._extract_methodology_details(text_to_scan)