# COMPREHENSIVE_SCIENTIFIC_REVIEW.md

This document provides an in-depth scientific rationale and literature review for all major methods, algorithms, and design choices in the workflow_16s pipeline. It is intended to support transparency, reproducibility, and scientific rigor for users, reviewers, and collaborators.

## 1. Upstream Processing
- **QIIME 2**: Chosen for its plugin-based, reproducible architecture and broad community validation ([Bolyen et al., 2019, Nat Biotechnol](https://www.nature.com/articles/s41587-019-0209-9)).
- **DADA2**: Used for denoising due to its model-based error correction and superior performance in benchmarking studies ([Callahan et al., 2016, Nat Methods](https://www.nature.com/articles/nmeth.3869)).
- **Deblur**: Available but not default, as it is less robust for highly diverse or large datasets ([Amir et al., 2017, mSystems](https://journals.asm.org/doi/10.1128/mSystems.00191-16)).
- **Primer/Region Filtering**: Optional, based on best practices for amplicon region specificity ([Klindworth et al., 2013, NAR](https://academic.oup.com/nar/article/41/1/e1/2904021)).

## 2. Downstream Analysis
- **AnnData/Scanpy**: Used for scalable, memory-efficient storage and analysis, leveraging the single-cell ecosystem for large microbiome datasets ([Wolf et al., 2018, Genome Biol](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0)).
- **Compositional Data Analysis**: CLR/ILR transformations and compositional-aware statistics are used to address the non-independence of amplicon data ([Gloor et al., 2017, mSystems](https://journals.asm.org/doi/10.1128/mSystems.00162-16)).
- **Multi-method DA**: DESeq2 ([Love et al., 2014, Genome Biol](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-014-0550-8)), ALDEx2 ([Fernandes et al., 2013, Microbiome](https://microbiomejournal.biomedcentral.com/articles/10.1186/2049-2618-1-23)), and Wilcoxon are combined for robust feature selection, as single-method DA is prone to false positives ([Weiss et al., 2017, Microbiome](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-017-0237-y)).
- **Batch Effect Correction**: ConQuR ([Wang et al., 2021, Nat Commun](https://www.nature.com/articles/s41467-021-22244-4)) is preferred for compositional data; ComBat ([Johnson et al., 2007, Biostatistics](https://academic.oup.com/biostatistics/article/8/1/118/252073)) is included for compatibility.
- **PERMANOVA, Kruskal-Wallis, Mann-Whitney U**: Standard nonparametric tests for community and feature-level comparisons ([Anderson, 2001, Austral Ecology](https://onlinelibrary.wiley.com/doi/abs/10.1046/j.1442-9993.2001.01070.x)).
- **Network Inference**: Correlation-based networks are default for speed and interpretability; SparCC ([Friedman & Alm, 2012, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687)) and SpiecEasi ([Kurtz et al., 2015, PNAS](https://www.pnas.org/doi/10.1073/pnas.1410504112)) are available for advanced users.
- **Decontamination**: Frequency/prevalence-based methods are used when negative controls are present ([Davis et al., 2018, Microbiome](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-018-0605-2)).

## 3. Machine Learning
- **CatBoost**: Chosen for its ability to handle categorical variables and strong performance on microbiome data ([Prokhorenkova et al., 2018, NIPS](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)).
- **SHAP**: Used for interpretable feature importance ([Lundberg & Lee, 2017, NIPS](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)).
- **Random Forests**: Used for both classification and regression, with OOB scoring and feature importance ([Breiman, 2001, Machine Learning](https://link.springer.com/article/10.1023/A:1010933404324)).
- **Cross-validation and Covariate Control**: Multiple strategies are compared to assess batch confounding and model robustness.

## 4. Visualization & Reporting
- **Plotly**: Used for interactive, high-quality figures; static plots are available for publication ([Plotly Technologies Inc., 2015](https://plotly.com/python/)).
- **Scanpy/Seaborn/Matplotlib**: Used for heatmaps, QC plots, and exploratory analysis.
- **HTML Reporting**: All results, diagnostics, and code provenance are included for transparency and reproducibility.

## 5. Methods Explicitly Avoided
- **LEfSe**: Not used due to poor batch effect handling and high false positive rate ([Nearing et al., 2022, Nat Microbiol](https://www.nature.com/articles/s41564-021-01007-0)).
- **Rarefaction**: Deprecated for normalization, as it discards data and reduces power ([McMurdie & Holmes, 2014, PLoS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003531)).
- **Single-method DA**: Avoided due to lack of robustness ([Weiss et al., 2017, Microbiome](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-017-0237-y)).
- **Manual Scripting**: All steps are automated and reproducible; ad hoc scripts are discouraged.

## 6. References
A full bibliography is maintained in the project docs. Key references are cited inline above for each method and design choice.

---

This review is updated regularly to reflect new best practices and scientific advances. For questions or suggestions, see the project README or contact the maintainers.
