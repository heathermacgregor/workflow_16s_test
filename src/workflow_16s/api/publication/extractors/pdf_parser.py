# workflow_16s/api/publication/extractors/pdf_parser.py

import io
import pdfplumber
from workflow_16s.utils.logger import get_logger

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_PDF_PAGES = 10

def fetch_and_parse_pdf(url, session):
    logger = get_logger("workflow_16s")
    # 1. Dress up like a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/pdf"
    }
                
    response = session.get(url, headers=headers, timeout=30)
                
    # Check if the request was successful
    if response.status_code != 200:
        logger.debug(f"[⚑] Failed to fetch {url} - Status Code: {response.status_code}")
        return None

    # 2. Verify we actually got a PDF back, not an HTML error page
    content_type = response.headers.get('Content-Type', '').lower()
    if 'application/pdf' not in content_type:
        logger.debug(f" ↠ Server returned {content_type} instead of a PDF for {url}. Skipping.")
        return None

    # Now it is safe to pass response.content to your PDF parser!
    return safely_extract_pdf(response.content, session)

def safely_extract_pdf(pdf_url, session) -> str | None: 
    logger = get_logger("workflow_16s")
    try:
        # 1. Make the request FIRST using stream=True
        with session.get(pdf_url, stream=True, timeout=30) as resp:
            # 2. Check for 403/401 errors
            if resp.status_code in [403, 401]:
                logger.debug(f" ↠ Access denied (HTTP {resp.status_code}) for {resp.url}. Publisher is blocking us.")
                return None
                            
            resp.raise_for_status() # Catch other HTTP errors
                        
            # 3. Check file size
            content_length = int(resp.headers.get('Content-Length', 0))
            if content_length > MAX_FILE_SIZE:
                logger.debug(f" ↠ Skipping PDF {pdf_url}: File too large ({content_length} bytes)")
                return None
                        
            # 4. Download the actual content now that checks passed
            pdf_content = resp.content
                        
            # 5. Check if it's actually a PDF by looking at the first few bytes
            # Real PDFs always start with '%PDF-'
            if not pdf_content.startswith(b"%PDF-"):
                logger.debug(f" ↠ URL {resp.url} returned HTML/text instead of a PDF. Skipping.")
                return None
                            
            # 6. Extract text with a strict page limit
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                pages_to_extract = pdf.pages[:MAX_PDF_PAGES]
                return "\\n".join(p.extract_text() for p in pages_to_extract if p.extract_text())
                
    except Exception as e:
        logger.debug(f"[⚑] Failed to extract PDF from {pdf_url}: {type(e).__name__} - {e}")
        return None