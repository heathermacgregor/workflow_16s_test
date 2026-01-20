# workflow_16s/downstream/utils/reporting.py

import pandas as pd
import datetime
from pathlib import Path
import logging

logger = logging.getLogger("workflow_16s")

def generate_synthesis_report(output_dir: Path):
    """
    Creates an executive summary HTML dashboard from various workflow outputs.
    Consolidates ML results, high-confidence biomarkers, and batch statistics.
    """
    logger.info("--- Generating Executive Synthesis Report ---")
    
    # 1. Load the Master Biomarker List
    biomarker_file = output_dir / "MASTER_BIOMARKER_SUMMARY.csv"
    if not biomarker_file.exists():
        logger.warning(f"Synthesis Report Skip: {biomarker_file.name} not found.")
        biomarker_html = "<p>Biomarker summary not available.</p>"
    else:
        df_bio = pd.read_csv(biomarker_file).head(20)
        biomarker_html = df_bio.to_html(classes='table table-striped', index=False)

    # 2. Check for Batch Auditing Results
    variance_file = output_dir / "statistical_analysis" / "variance_partitioning.html"
    variance_link = f"<a href='{variance_file}'>View Detailed Variance Partitioning</a>" if variance_file.exists() else "Not Run"

    # 3. Build HTML Template
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workflow 16S Synthesis Report</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
        <style>
            body {{ padding: 40px; background-color: #f8f9fa; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ margin-top: 30px; color: #2980b9; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>16S Downstream Analysis: Executive Summary</h1>
            <p class="text-muted">Generated on: {now}</p>
            
            <section>
                <h2>Top 20 Batch-Invariant Biomarkers</h2>
                <p>These microbial indicators showed high stability across the 300+ sequencing batches.</p>
                {biomarker_html}
            </section>

            <section>
                <h2>Batch Effect & Technical Audit</h2>
                <p>Quantification of technical variance (Batch ID) vs Biological variance (Facility).</p>
                <ul>
                    <li>Variance Partitioning Status: {variance_link}</li>
                </ul>
            </section>

            <div class="footer text-center">
                Workflow 16S | Automated Downstream Synthesis Module
            </div>
        </div>
    </body>
    </html>
    """
    
    # 4. Save to disk
    report_path = output_dir / "EXECUTIVE_SUMMARY.html"
    try:
        with open(report_path, "w") as f:
            f.write(html_content)
        logger.info(f"✅ Executive Dashboard saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write synthesis report: {e}")
        
def summarize_metabolic_blocks(adata, output_dir):
    """Groups individual EC/Pathways into high-level metabolic blocks."""
    if 'picrust2_pathways' not in adata.obsm:
        return
    
    pathway_df = pd.DataFrame(adata.obsm['picrust2_pathways'], index=adata.obs_names)
    
    # Define metabolic keywords for grouping
    blocks = {
        'Carbon Metabolism': ['carbon', 'glycolysis', 'cycle', 'sugar'],
        'Nitrogen Cycling': ['nitrogen', 'nitrate', 'ammonia', 'denitrif'],
        'Sulfur/Xenobiotics': ['sulfur', 'degrad', 'toluene', 'benzoate'],
        'Stress/Defense': ['antibiotic', 'resistance', 'stress', 'oxidative']
    }
    
    summary_results = {}
    for block_name, keywords in blocks.items():
        pattern = '|'.join(keywords)
        matched_cols = [c for c in pathway_df.columns if any(re.search(pattern, c, re.I))]
        if matched_cols:
            summary_results[block_name] = pathway_df[matched_cols].mean(axis=1)
    
    block_df = pd.DataFrame(summary_results)
    block_df.to_csv(output_dir / "functional_metabolic_blocks.csv")
    logger.info(f"Summarized {len(blocks)} metabolic blocks for reporting.")