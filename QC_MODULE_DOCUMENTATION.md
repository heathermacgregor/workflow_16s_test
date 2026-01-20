# Quality Control (QC) Module Documentation

**Date:** January 7, 2026  
**Version:** 2.0+

## Overview

The `workflow_16s.qc` module provides state-of-the-art quality control for 16S rRNA amplicon sequencing data, addressing critical data integrity issues in microbial community analysis.

## Key Features

### 1. **Metadata Validation** (`MetadataValidator`)
- ✅ Automatic redundancy removal (duplicate/correlated columns)
- ✅ Numeric range validation with environment-specific thresholds
- ✅ ENVO ontology integration for semantic categorization
- ✅ Unit harmonization (temperature, depth, coordinates)
- ✅ External data validation (SoilGrids, Meteostat, facility data)
- ✅ Geographic consistency checking

### 2. **ENVO Semantic Categorization** (`ENVOOntology`)
- ✅ Find all "soil" samples even with variations:
  - "soil [ENVO:00001998]"
  - "forest soil"
  - "agricultural soil"
  - Different languages/synonyms
- ✅ Hierarchical biome classification
- ✅ Confidence scoring

### 3. **Primer Quality Control** (`PrimerQC`)
- ✅ Multi-pass primer detection (5', 3', anywhere)
- ✅ Mismatch tolerance using CutAdapt
- ✅ Orientation analysis (forward/reverse/mixed)
- ✅ Adapter contamination detection (Illumina, Nextera, PhiX)
- ✅ Trimming validation (before/after comparison)
- ✅ Per-sample and aggregate reporting

### 4. **Sample Identity Validation** (`SampleIdentityValidator`)
- ✅ Cross-validate claimed vs. observed environment types
- ✅ Human contamination detection in environmental samples
- ✅ Primer region vs. ASV length matching
- ✅ Metadata outlier detection
- ✅ Flags mislabeled samples

### 5. **Enhanced Contamination Detection** (`contamination_enhanced`)
- ✅ Reference database matching (kitome, human taxa)
- ✅ Frequency-based detection (no controls needed!)
- ✅ Ubiquity-based detection (low abundance + high prevalence)
- ✅ Cross-sample contamination detection
- ✅ Works on public data without negative controls

## Quick Start

### Simplest Usage

```python
from workflow_16s.qc import quick_qc

# Run all QC steps with one command
adata_clean = quick_qc(
    adata, 
    output_dir='qc_results',
    remove_contaminants=True
)
```

### Custom Usage

```python
from workflow_16s.qc import ComprehensiveQC

# Initialize QC pipeline
qc = ComprehensiveQC(config={
    'primer_max_error_rate': 0.15,
    'primer_max_reads': 10000,
    'n_cores': 8
})

# Run complete pipeline
adata_clean, results = qc.run_all(
    adata,
    fastq_files=['sample1_R1.fastq.gz', 'sample2_R1.fastq.gz'],
    primers={'V4_fwd': 'GTGCCAGCMGCCGCGGTAA', 'V4_rev': 'GGACTACHVGGGTWTCTAAT'},
    output_dir='qc_results',
    remove_contaminants=True
)
```

## Individual Module Usage

### 1. Metadata Validation

```python
from workflow_16s.qc import MetadataValidator

# Initialize validator
validator = MetadataValidator(metadata_df, config={})

# Run all validation steps
cleaned_metadata, validation_report = validator.validate_all()

# Save reports
validation_report.to_csv('metadata_validation_report.csv', index=False)
```

**What it does:**
- Removes redundant columns (e.g., keeps `latitude` over `latitude_ena`)
- Removes highly correlated columns (r > 0.99)
- Validates pH (0-14), temperature (-50 to 120°C), salinity, coordinates
- Harmonizes units (Kelvin → Celsius, cm → meters)
- Adds semantic categorization (env_category_biome, env_category_type)

**Output example:**
```
Initial columns: 247
Removed 89 redundant columns
Final columns: 158

Validation issues:
- 3 samples with pH outside valid range [0-14]
- 15 samples with temperature outside typical range [-10, 50°C]
- 7 samples with low-confidence ENVO categorization
```

### 2. ENVO Semantic Categorization

```python
from workflow_16s.qc import ENVOOntology

envo = ENVOOntology()

# Find ALL soil samples (semantic search)
soil_samples = envo.find_samples_by_category(
    metadata_df,
    target_category='soil',
    min_confidence=0.5
)

# Categorize a single sample
category = envo.categorize_sample(
    env_biome='terrestrial biome [ENVO:00000446]',
    env_feature='forest soil [ENVO:00002259]',
    env_material='soil [ENVO:00001998]'
)
# Returns: {
#   'biome_class': 'terrestrial',
#   'category': 'soil',
#   'material_type': 'solid',
#   'confidence': 0.9,
#   'matched_terms': ['soil', 'forest soil', 'ENVO:00001998']
# }
```

**Supported categories:**
- **Terrestrial:** soil, sediment_terrestrial
- **Aquatic:** marine, freshwater, sediment_aquatic
- **Extreme:** hot_spring, hypersaline, permafrost, acid_mine
- **Built:** wastewater, bioreactor, composting

### 3. Primer QC

```python
from workflow_16s.qc import PrimerQC

# Initialize
primer_qc = PrimerQC(
    primers={
        'V4_fwd': 'GTGCCAGCMGCCGCGGTAA',
        'V4_rev': 'GGACTACHVGGGTWTCTAAT'
    },
    max_error_rate=0.15,  # 15% mismatch tolerance
    min_overlap=10,
    max_reads=10000
)

# Check single file
report = primer_qc.comprehensive_check('sample_R1.fastq.gz')

# Batch check multiple files
results = primer_qc.batch_check(
    fastq_files=['sample1_R1.fastq.gz', 'sample2_R1.fastq.gz'],
    output_report='primer_qc_report.html'
)

# Validate trimming worked
validation = primer_qc.validate_trimming(
    'pre_trim.fastq.gz',
    'post_trim.fastq.gz'
)
```

**Report includes:**
- % reads with primers detected (should be >80%)
- Orientation (forward/reverse/mixed)
- Contamination (Illumina adapters, PhiX, other primers)
- Length distribution before/after
- Overall assessment: PASS/WARNING/FAIL

### 4. Sample Identity Validation

```python
from workflow_16s.qc import SampleIdentityValidator

# Initialize
validator = SampleIdentityValidator(adata)

# Run all validation checks
validation_results = validator.validate_all()

# Get flagged samples
flagged = validator.get_flagged_samples(min_severity='WARNING')

# Generate report
validator.generate_report('sample_validation_report.csv')
```

**Checks performed:**
1. **Environment Type:** Do observed taxa match claimed environment?
   - Claimed: "soil" → Expected: Proteobacteria, Actinobacteria, Acidobacteria
   - If 90% human gut taxa found → FLAG as mislabeled
   
2. **Human Contamination:** Detect skin/gut/oral taxa in environmental samples
   - NONE: <5% human taxa
   - LOW: 5-15%
   - MEDIUM: 15-30%
   - HIGH: >30% (likely contaminated or mislabeled)

3. **Primer Region:** ASV length matches claimed region
   - V4: 240-280bp
   - V3-V4: 420-480bp
   - Flags if median length differs >50bp

4. **Metadata Outliers:** Samples very different from their group

**Output:**
```
Sample Validation Summary:
PASS: 847 samples (84.7%)
WARNING: 89 samples (8.9%)
FAIL: 64 samples (6.4%)

Recommended action: Investigate FAIL samples, consider excluding
```

### 5. Enhanced Contamination Detection

```python
from workflow_16s.qc import detect_contaminants_reference_based

# Run detection (no negative controls needed!)
contam_scores = detect_contaminants_reference_based(
    adata,
    method='combined',  # database + frequency + ubiquity
    prevalence_threshold=0.9,
    abundance_threshold=0.001,
    exclude_env_types=['gut']  # Don't flag human taxa in gut samples
)

# Features flagged as contaminants
contaminants = contam_scores[contam_scores['is_contaminant']]

# Remove contaminants
from workflow_16s.qc import remove_contaminants_enhanced
adata_clean = remove_contaminants_enhanced(
    adata,
    contam_scores,
    threshold=0.5,
    inplace=False
)
```

**Detection methods:**

1. **Database matching:** Known contaminants from literature
   - Kitome (DNA extraction kits): Bradyrhizobium, Sphingomonas, etc.
   - Human skin: Propionibacterium, Staphylococcus, etc.
   - Human gut: Bacteroides, Prevotella, etc.
   - Score: 0-1 based on confidence

2. **Frequency-based:** Taxa inversely correlated with total reads
   - Low-biomass samples = more contamination
   - High-biomass samples = less contamination
   - Real taxa: positive or no correlation
   - Contaminants: negative correlation

3. **Ubiquity-based:** "Kitome signature"
   - Present in >90% samples
   - Low abundance (<0.1%)
   - Consistent across samples
   - Typical of reagent contaminants

4. **Combined:** Consensus scoring from all methods
   - More robust than single method
   - Reduces false positives

## Output Files

When using `ComprehensiveQC` or `quick_qc` with `output_dir` specified:

```
qc_results/
├── metadata_validation_report.csv  # Metadata validation issues
├── sample_validation_report.csv    # Sample identity validation
├── contamination_scores.csv        # Contamination scores per feature
├── primer_qc_results.csv           # Primer detection stats (if FASTQ provided)
├── qc_summary.csv                  # Overall summary statistics
└── qc_report.html                  # Comprehensive HTML report
```

## Integration with Downstream Workflow

The QC results are automatically added to `adata.obs`:

```python
# After running QC
print(adata.obs.columns)
# Includes:
# - env_category_biome
# - env_category_type
# - env_category_confidence
# - qc_env_match
# - qc_human_contamination
# - qc_primer_match
# - qc_metadata_outlier
# - qc_overall_flag
```

Filter samples based on QC:

```python
# Keep only PASS samples
adata_pass = adata[adata.obs['qc_overall_flag'] == 'PASS', :]

# Or keep PASS + WARNING
adata_filtered = adata[adata.obs['qc_overall_flag'] != 'FAIL', :]
```

## Example: Complete Workflow with QC

```python
from workflow_16s.qc import ComprehensiveQC
from workflow_16s.downstream import DownstreamWorkflow

# 1. Run comprehensive QC
qc = ComprehensiveQC()
adata_clean, qc_results = qc.run_all(
    adata,
    output_dir='qc_results',
    remove_contaminants=True
)

# 2. Filter to high-quality samples
adata_hq = adata_clean[adata_clean.obs['qc_overall_flag'] == 'PASS', :]

# 3. Continue with downstream analysis
workflow = DownstreamWorkflow(config)
workflow.adata = adata_hq
workflow.run_diversity_analysis()
workflow.run_differential_abundance()
```

## Best Practices

1. **Always run metadata validation first**
   - Catches issues early
   - Cleans data for downstream analyses
   - Adds semantic categorization

2. **Use ENVO categorization for filtering**
   ```python
   # Get all soil samples (semantic search)
   soil_mask = adata.obs['env_category_type'] == 'soil'
   soil_adata = adata[soil_mask, :]
   ```

3. **Review flagged samples before removing**
   ```python
   # Check what was flagged
   flagged = adata.obs[adata.obs['qc_overall_flag'] == 'FAIL']
   print(flagged[['env_biome', 'qc_env_match', 'qc_human_contamination']])
   ```

4. **Use combined contamination detection**
   - More robust than single method
   - Works without negative controls

5. **Check primer QC before blaming biology**
   - If differential abundance analysis finds nothing, check primer detection
   - Low primer detection = bad data quality

## Troubleshooting

### Metadata Validation

**Issue:** Too many columns removed
- **Solution:** Check `validation_report` for which columns were redundant
- Set custom thresholds for correlation in config

**Issue:** Many coordinate validation errors
- **Solution:** Check if coordinates are in different units (degrees vs. decimal)
- Run `harmonize_units()` separately first

### Primer QC

**Issue:** CutAdapt not found
```bash
conda install -c bioconda cutadapt
```

**Issue:** Low primer detection (<50%)
- Wrong primers specified
- Primers already trimmed
- Degraded library
- Wrong sequencing platform

### Sample Validation

**Issue:** Many samples flagged as FAIL
- Check if environment types are correctly assigned
- May have genuine human contamination
- May have mislabeled samples in database

### Contamination Detection

**Issue:** Too many features flagged
- Lower threshold (e.g., 0.7 instead of 0.5)
- Check if samples are actually low-biomass
- Review flagged genera - may be real

## References

1. **decontam:** Davis et al. 2018, Microbiome
2. **Kitome contaminants:** Salter et al. 2014, BMC Biology; Eisenhofer et al. 2019, mSystems
3. **ENVO ontology:** Buttigieg et al. 2016, J Biomed Semantics
4. **CutAdapt:** Martin 2011, EMBnet.journal

## Support

For issues or questions:
1. Check `qc_report.html` for detailed diagnostics
2. Review CSV reports for specific samples
3. See main pipeline documentation: `README_NEW_FEATURES.md`
