# History and Cross-Disciplinary Trends in the Biomedical, Life and Social Sciences 
This repository contains the code and analyses for a forthcoming paper title 'Historical and cross-disciplinary trends in the biomedical, life and social sciences reveal a shift from classical to multivariate statistics'. 

## Article Abstract
Methods for data analysis in the biomedical, life and social sciences are developing at a rapid pace. At the same time, there is increasing concern that education in quantitative methods is failing to adequately prepare students for contemporary research. These trends have led to calls for educational reform to undergraduate and graduate quantitative research method curricula. We argue that such reform should be based on data-driven insights into within- and cross-disciplinary method usage. Our survey of peer-reviewed literature screened ~3.5 million openly available research articles to monitor the cross-disciplinary usage of research methods in the past decade. We applied data-driven text-mining analyses to the methods and materials section of a large subset of this corpus to identify method trends shared across disciplines, as well as those unique to each discipline. As a whole, usage of t-test, analysis of variance, and other classical regression-based methods has declined in the published literature over the past 10 years. Machine-learning approaches, such as artificial neural networks, have seen a significant increase in the total share of scientific publications. We find unique groupings of research methods associated with each biomedical, life and social science discipline, such as the use of structural equation modeling in psychology, survival models in oncology, and manifold learning in ecology. We discuss the implications of these findings for education in statistics and research methods, as well as cross- and trans-disciplinary collaboration. 

## Getting Started
* Note, some of these steps are only necessary to replicate the results in the study. If you're interested in specific components of the preprocessing and analysis pipeline, some of these steps are not necessary.

### 1. Installing
```
git clone https://github.com/tsb46/stats_history.git
```
Then, in the base directory install all necessary packages (in requirements.txt):
```

```

### 2. Download all Pubmed Open Access Subset and Author Manuscript Collection XML files from the Pubmed FTP Service (https://ftp.ncbi.nlm.nih.gov/pub/pmc/).
* Pubmed Open Access Subset files: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
* Author Manuscript Collection files: https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/
* These files are VERY large, and will take a while to download!


```

```
