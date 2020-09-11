# Historical and Cross-Disciplinary Trends in the Biomedical, Life and Social Sciences 
This repository contains the code and analyses for a forthcoming paper title 'Historical and cross-disciplinary trends in the biomedical, life and social sciences reveal a shift from classical to multivariate statistics'. 

## Article Abstract
Methods for data analysis in the biomedical, life and social sciences are developing at a rapid pace. At the same time, there is increasing concern that education in quantitative methods is failing to adequately prepare students for contemporary research. These trends have led to calls for educational reform to undergraduate and graduate quantitative research method curricula. We argue that such reform should be based on data-driven insights into within- and cross-disciplinary method usage. Our survey of peer-reviewed literature screened ~3.5 million openly available research articles to monitor the cross-disciplinary usage of research methods in the past decade. We applied data-driven text-mining analyses to the methods and materials section of a large subset of this corpus to identify method trends shared across disciplines, as well as those unique to each discipline. As a whole, usage of t-test, analysis of variance, and other classical regression-based methods has declined in the published literature over the past 10 years. Machine-learning approaches, such as artificial neural networks, have seen a significant increase in the total share of scientific publications. We find unique groupings of research methods associated with each biomedical, life and social science discipline, such as the use of structural equation modeling in psychology, survival models in oncology, and manifold learning in ecology. We discuss the implications of these findings for education in statistics and research methods, as well as cross- and trans-disciplinary collaboration. 

# Getting Started

## 1. Repo Directory Structure
```
|-- ./pubmed_query.py
|-- ./pubmed_xml_parse.py
|-- ./ner_trainer
|   |-- ./ner_trainer/__init__.py
|   |-- ./ner_trainer/user_prompts.py
|   |-- ./ner_trainer/trainer_utils.py
|   |-- ./ner_trainer/phrase_seeds.txt
|   |-- ./ner_trainer/ner_interactive_trainer.py
|   |-- ./ner_trainer/spacy_train.py
|   |-- ./ner_trainer/ner_automatic_trainer.py
|   |-- ./ner_trainer/ner_spacy_model.pickle
|   `-- ./ner_trainer/extract_ents.py
|-- ./analysis_main
|   |-- ./analysis_main/utils.py
|   |-- ./analysis_main/ent_classification.csv
|   |-- ./analysis_main/ents_trends.py
|   |-- ./analysis_main/ents_base.py
|   |-- ./analysis_main/ents_compare.py
|   |-- ./analysis_main/ents_domain.py
|   |-- ./analysis_main/ents_network.py
|   `-- ./analysis_main/preprocessing.py
|-- ./demo.ipynb
|-- ./requirements.txt
|-- ./journal_classification
|   |-- ./journal_classification/generate_classification_training_samples.py
|   |-- ./journal_classification/journal_classification_training_all.csv
|   |-- ./journal_classification/custom_transformers.py
|   |-- ./journal_classification/__pycache__
|   |   |-- ./journal_classification/__pycache__/train_naive_bayes_classifier.cpython-37.pyc
|   |   `-- ./journal_classification/__pycache__/custom_transformers.cpython-37.pyc
|   |-- ./journal_classification/train_naive_bayes_classifier.py
|   `-- ./journal_classification/nb_classifier.pickle
|-- ./README.md

```
* <b>Demo.ipynb</b>: Jupyter notebook illustrating some of the results from the study
* <b>ner_trainer</b>: directory containing a set of command-line utilities for training the Named Entity Recognition model to detect statistical method 'entities' in article 'Methods and Materials' sections. This directory also contains utility functions for creating automatic training samples directly from a list of example strings - via the 'phrase_seeds.txt'. The fully-trained NER model used in the paper is also contained in this directory as a pickle file: 'ner_spacy_model.pickle'.
* <b>journal classification</b>: directory containing a set of command-line utilities for training the Multinomial Naive Bayes algorithm for classifying articles into the 15 scientific disciplines described in the paper. The fully-trained classifier model used in the paper is contained in this directory as a pickle file: 'nb_classifier.pickle'.
* <b>analysis_main</b>: directory containing all the analysis code for the paper. The three analyses as described in the paper: 1) method usage trends, 2) discipline by research method probability analysis and 3) analysis of research method groupings. 

## 2. Installing
* Note, some of these steps are only necessary to replicate the results in the study. If you're interested in specific components of the preprocessing and analysis pipeline, some of these steps are not necessary.
All of the code in this repo was run with Python 3.7. Not 100% sure code will work with other versions of Python 3.
Assuming Git has already been installed on your desktop, copy and paste the following code into your terminal to download the repo:
```
git clone https://github.com/tsb46/stats_history.git
```
Then, in the base directory of the repo ('stats_history'), pip install all necessary packages (in requirements.txt) with the following command in the terminal:
```
pip install -r requirements.txt
```
In order to run the text preprocessing described in the paper, you will need to download some data from NLTK and download one of SpaCy's english language models:
#### NLTK Downloads 
In the Python interpreter:
```
import nltk
nlkt.download('punkt') # For NLTK word tokenizer
nltk.download('wordnet') # For NLTK WordNet Lemmatizer
```
#### SpaCy Model Download
In the terminal:
```
python -m spacy download en_core_web_sm
```
## 3. Download and Parse Full-Text Articles
### Download Articles
The Pubmed Open Access Subset and Author Manuscript Collection XML files can be downloaded from Pubmed FTPs service:
* Pubmed Open Access Subset files: https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
* Author Manuscript Collection files: https://ftp.ncbi.nlm.nih.gov/pub/pmc/manuscript/
* These files are VERY large, and will take a while to download!
### Parsing Full-Text Articles





```

```
