# ==================================================================================== #
# NEW FILE: workflow_16s/downstream/run_conqur.R
# ==================================================================================== #

# --- Auto-install Missing Packages ---
# List of required packages
required_packages <- c("argparse", "ConQuR", "doParallel", "quantreg", "cqrReg", "glmnet", 
                     "dplyr", "gplots", "vegan", "ade4", "compositions", 
                     "randomForest", "ROCR", "ape", "GUniFrac", "fastDummies")

# Install devtools if not present
if (!requireNamespace("devtools", quietly = TRUE)) {
    install.packages("devtools", repos = "http://cran.us.r-project.org")
}

# Install ConQuR from GitHub if not present
if (!requireNamespace("ConQuR", quietly = TRUE)) {
    print("Installing ConQuR from GitHub...")
    devtools::install_github("wdl2459/ConQuR")
}

# Install other missing packages from CRAN
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
    print(paste("Installing missing packages:", paste(new_packages, collapse=", ")))
    install.packages(new_packages, repos = "http://cran.us.r-project.org")
}

# --- Load Libraries ---
suppressPackageStartupMessages({
    library(argparse)
    library(ConQuR)
    library(doParallel)
    library(dplyr)
    library(compositions) # for CLR
    library(vegan) # for PERMANOVA
})

# --- Define Command Line Arguments ---
parser <- ArgumentParser(description='Run ConQuR Batch Correction from Python')
parser$add_argument("-i", "--input_counts", type="character", required=TRUE, help="Path to the raw count table (TSV)")
parser$add_argument("-m", "--input_metadata", type="character", required=TRUE, help="Path to the metadata table (TSV)")
parser$add_argument("-o", "--output_counts", type="character", required=TRUE, help="Path to save the corrected count table (TSV)")
parser$add_argument("-b", "--batch_col", type="character", required=TRUE, help="Name of the batch column in the metadata")
parser$add_argument("-k", "--key_vars", type="character", required=TRUE, help="Comma-separated list of key biological variables to preserve (e.g., 'disease_status,age')")
parser$add_argument("-c", "--covariates", type="character", default=NULL, help="Optional comma-separated list of other covariates (e.g., 'sex,bmi')")

args <- parser$parse_args()

options(warn = -1) # Suppress warnings from quantile regression
print("ConQuR packages loaded.")

# --- Load Data ---
print(paste("Loading counts:", args$input_counts))
# The count table from anndata will have samples as rows, taxa as columns.
taxa <- read.csv(args$input_counts, sep="\t", row.names=1, check.names=FALSE)

print(paste("Loading metadata:", args$input_metadata))
meta <- read.csv(args$input_metadata, sep="\t", row.names=1, check.names=FALSE)

# Align data
shared_samples <- intersect(rownames(taxa), rownames(meta))
if (length(shared_samples) == 0) {
    stop("Error: No common samples found between count table and metadata.")
}
taxa <- taxa[shared_samples, ]
meta <- meta[shared_samples, ]
print(paste("Aligned data. Found", length(shared_samples), "samples in common."))

# --- Prepare Data for ConQuR ---
# 1. Batch ID (must be a factor)
if (!args$batch_col %in% colnames(meta)) {
    stop(paste("Error: Batch column", args$batch_col, "not found in metadata."))
}
batchid <- factor(meta[[args$batch_col]])
print(paste("Using batch column:", args$batch_col))
print("Batch summary:")
print(summary(batchid))

# 2. Covariates (key variables + other covariates)
all_covar_names <- strsplit(args$key_vars, ",")[[1]]
if (!is.null(args$covariates) && nchar(args$covariates) > 0) {
    other_covars <- strsplit(args$covariates, ",")[[1]]
    all_covar_names <- c(all_covar_names, other_covars)
}
all_covar_names <- unique(all_covar_names) # Remove duplicates

# Check if all covar names are in metadata
missing_covars <- all_covar_names[!(all_covar_names %in% colnames(meta))]
if (length(missing_covars) > 0) {
    print(paste("Warning: The following covariates were not found in the metadata and will be skipped:", paste(missing_covars, collapse=", ")))
    all_covar_names <- all_covar_names[all_covar_names %in% colnames(meta)]
}

if (length(all_covar_names) == 0) {
    stop("Error: No valid key variables or covariates found to preserve. ConQuR requires at least one covariate.")
}

print(paste("Preserving variables:", paste(all_covar_names, collapse=", ")))
covar <- meta[, all_covar_names, drop=FALSE]

# Convert character columns in covar to factors (as required by ConQuR)
covar <- as.data.frame(unclass(covar), stringsAsFactors = TRUE)

# --- Run Tune_ConQuR ---
# This is the more robust approach. It will test different batches
# as the reference and select the one that minimizes the batch effect.
print("Running Tune_ConQuR to find the optimal reference batch... (This may take a long time)")

# We will test all batches as potential references
batch_ref_candidates <- levels(batchid)

# Define the tuning parameters
# We will test the default ("standard") vs. the "penalized" (lasso) strategies
# We will test taxa with prevalence from 0% to 100%
result_tuned <- Tune_ConQuR(
    tax_tab = taxa,
    batchid = batchid,
    covariates = covar,
    batch_ref_pool = batch_ref_candidates,
    logistic_lasso_pool = c(FALSE, TRUE),
    quantile_type_pool = c("standard", "lasso"),
    simple_match_pool = c(FALSE),
    lambda_quantile_pool = c(NA, "2p/n"),
    interplt_pool = c(FALSE, TRUE),
    frequencyL = 0.0,
    frequencyU = 1.0,
    cutoff = 0.1, # Divide taxa into 10 prevalence groups for tuning
    num_core = 2
)

print("ConQuR tuning complete. Using optimal corrected table.")
taxa_corrected <- result_tuned$tax_final

# --- Save Corrected Data ---
# The output is a matrix. Convert back to data.frame to save with headers.
taxa_corrected_df <- as.data.frame(taxa_corrected)
write.table(taxa_corrected_df, file=args$output_counts, sep="\t", quote=FALSE, col.names=NA)

print(paste("Corrected count table saved to:", args$output_counts))
print("R script finished.")