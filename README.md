# Historical and Cross-Disciplinary Trends in the Biomedical, Life and Social Sciences 
This repository contains the code and analyses for a forthcoming paper title 'Historical and cross-disciplinary trends in the biomedical, life and social sciences reveal a shift from classical to multivariate statistics'. 

## Article Abstract
Methods for data analysis in the biomedical, life and social sciences are developing at a rapid pace. At the same time, there is increasing concern that education in quantitative methods is failing to adequately prepare students for contemporary research. These trends have led to calls for educational reform to undergraduate and graduate quantitative research method curricula. We argue that such reform should be based on data-driven insights into within- and cross-disciplinary use of analytic methods. Our survey of peer-reviewed literature analyzed ~1.3 million openly available research articles to monitor the cross-disciplinary mentions of analytic methods in the past decade. We applied data-driven text-mining analyses to the methods and results sections of a large subset of this corpus to identify trends in analytic method mentions shared across disciplines, as well as those unique to each discipline. We found that T-test, analysis of variance, linear regression, chi-squared test and other classical statistical methods have been, and remain the most mentioned analytic methods in biomedical, life science and social science research articles. However, mentions of these methods have declined as a percentage of the published literature over the past decade. On the other hand, multivariate statistical and machine-learning approaches, such as artificial neural networks, have seen a significant increase in the total share of scientific publications. We also find unique groupings of analytic methods associated with each biomedical, life and social science discipline, such as the use of structural equation modeling in psychology, survival models in oncology, and manifold learning in ecology. We discuss the implications of these findings for education in statistics and research methods, as well as within- and cross-disciplinary collaboration. 

# Getting Started

## 1. Repo Directory Structure
```
├── README.md
├── analysis_main
│   ├── ent_classification.csv
│   ├── ents_base.py
│   ├── ents_compare.py
│   ├── ents_domain.py
│   ├── ents_network.py
│   ├── ents_trends.py
│   └── prerun_model_dict.pickle
├── demo.ipynb
├── entity_preprocessing
│   ├── __pycache__
│   │   ├── preprocessing.cpython-38.pyc
│   │   └── utils.cpython-38.pyc
│   ├── preprocess_entities.py
│   ├── preprocessing.py
│   └── utils.py
├── journal_classification
│   ├── classify_articles.py
│   ├── custom_transformers.py
│   ├── generate_classification_training_samples.py
│   ├── journal_classification_training_all.csv
│   ├── nb_classifier.pickle
│   └── train_naive_bayes_classifier.py
├── ner_extraction
│   ├── __init__.py
│   ├── extract_ents.py
│   ├── generate_training_samples.py
│   ├── ner_spacy_model.pickle
│   ├── phrase_seeds.txt
│   ├── section_markers.txt
│   ├── spacy_train.py
│   └── trainer_utils.py
├── pubmed_query.py
├── requirements.txt
├── results
└── data

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
* The files are organized into directories as follows (as of 04/06/21):
```
├── PMC001xxxxxx
├── PMC002xxxxxx
├── PMC003xxxxxx
├── PMC004xxxxxx
├── PMC005xxxxxx
├── PMC006xxxxxx
├── PMC007xxxxxx
├── comm_use.A-B.xml
├── comm_use.C-H.xml
├── comm_use.I-N.xml
├── comm_use.O-Z.xml
├── non_comm_use.A-B.xml
├── non_comm_use.C-H.xml
├── non_comm_use.I-N.xml
└── non_comm_use.O-Z.xml
```
* We placed all the directories (and sub-directories) in the 'data' directory for further analysis.

# Article Parsing and Named Entity Recognition
All article parsing and NER tasks are contained in the 'ner_extraction' directory. **All command-line utilities in this directory are meant to be run from the base directory.**

## 1. Article Parsing Summary
We use the PubMed parser package (https://github.com/titipata/pubmed_parser) to extract article text and metadata from all XML files. The article text field of each article's XML file are divided into headings corresponding to the section title.. As we were only interested in extracting analytic methods mentions from methodology and results section text, we used a manually created list of section phrases containing search strings common for methodology and results section headers (e.g. 'methods', 'materials', 'results', 'analysis'). These section header search phrases are contained in 'section_markers.txt'. 

## 2. Training NER Parser
### 2.1 Generating Training Samples
We trained the NER algorithm by generating training samples from a fixed-phrase list contained in 'phrase_seeds.txt'. The phrases in this list (N=1129) were manually created by the authors to capture a wide-variety of potential analytic methods used in the biological, life and social sciences. For named entity recognition of analytic methods, we utilized the NER algorithm in SpaCy (https://github.com/explosion/spaCy). Training data for the SpaCy algorithm is generated using the command line utility 'generate_training_samples.py'. We used the following terminal command to generate 20,000 randomly selected training samples (documentation of each parameter is provided in the file).

In the terminal:
```
python ner_extraction/generate_training_samples.py -i data -o results -n 20000 -e STAT -s phrase_seeds.txt -k section_markers.txt
```
The above code will write the random sample of training data to a pickle file in the 'results' directory.

### 2.2 Training NER Algorithm
Using the training data generated from the fixed phrase list, we trained the SpaCy NER algorithm using a command-line utility in the 'spacy_train.py' script. We used the following command to train the SpaCy model (assuming the training data was written out to the 'results' directory):

In the terminal:
```
python ner_extraction/spacy_train.py -i results/training_samples_data.pickle -o ner_extraction
```

The above code will write the trained model to a pickle file for further use. **For those interested in the original trained SpaCy model used in the paper, we have provided the pretrained model: 'ner_extraction/ner_spacy_model.pickle'.** 

## 3. Article Parsing and Analytic Methods Extraction
With the pretrained NER model (either trained by the user, or using the pretrained file already provided), we parse and extract analytic methods from the corpus of research articles using the following command-line utility: 'extract_ents.py'. We created a sub-directory within the 'results' directory named 'ents' to write the results of the parsing and extraction step, and further steps below. Within this sub-directory we created a sub-directory 'original' to write articles parsing and extraction step. Each research article is parsed individually, which opens up opportunities for parallel computation. One may use the '-s' (start_indx) and '-e' (end_indx) to parse a select range of files. Using these parameters one may distribute the parsing and extraction of article batches across multiple cores/threads. The following command was used to parse articles and extract analytic method entities (assuming no parallel computation):

In the terminal:
```
python ner_extraction/extract_ents.py -i data -o results/ents/original -m ner_extraction/ner_spacy_model.pickle -k ner_extraction/section_markers.txt  
```
# Article Classification into Disciplines
Along with the extraction of analytic methods from research article methodology and results sections, each article was classified into one of 15 disciplines:
*  animal/insect/plant biology (ANIMAL)
*  biochemistry and molecular biology (BIOCHEM) 
*  clinical research (CLINIC) 
*  computer science and informatics (CS) 
*  ecology and evolutionary science (ECO) 
*  oncology (ONCO)
*  environmental science (ENVIRON) 
*  psychology (PSYCH) 
*  population and behavioral genetics (POPGENE) 
*  neuroscience (NEURO) 
*  chemistry and material science (CHEM) 
*  engineering and biotechnology (ENG) 
*  human physiology (PHYSIO) 
*  immunology (IMMUN)
*  epidemiology and public health (EPIDEM)

We input a bag-of-words feature-set (1-gram, 2-gram and 3-gram tokens), generated from the article abstract, title, and journal title, to a multinomial naïve Bayes (MNB) classification algorithm. We utilized a version of the MNB algorithm in the scikit-learn package (https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) that corrects for an unequal number of instances across categories, as was present in our corpus. To reduce the number of features input to the MNB algorithm and improve prediction accuracy we applied a chi-square feature selection approach. Specifically, we chose the top 15,000 features from the bag-of-word feature-set, where each feature was sorted by their chi-square statistic value with the 15 discipline categories. The chi-square statistic is a measure of dependence between two categorical variables. The final MNB model for training involved 15,000 features, 15 discipline categories to be predicted, and 1,470 training samples. We assessed the model accuracy using a repeated K-fold cross-validation approach (#folds = 10, # of repeats = 10). 

We have provided the training examples used in the paper to train our classifier in 'journal_classification_training_all.csv'. The pretrained classifier model used in the paper is provided in 'nb_classifier.pickle'. We encourage users to try their own discipline categorization, and have provided a command-line utility to retrain the model in 'train_naive_bayes_classifier.py'. As long as the training examples for a new discipline categorization follow the format of that in the provided training examples .csv, the classifier script should work appropriately. Parameters are provided for the repeated K-Fold cross-validation in the training script, such as the number of folds (-nf), the number of repeats (-nr) and the number of features to select from the chi-square feature selection (-nfeat). To retrain the classifier model, use the command:

In the terminal:
```
python journal_classification/train_naive_bayes_classifier.py -i journal_classification/journal_classification_training_all.csv -o journal_classification  
```

To classify articles for further processing using the pretrained classifier model, we created a command-line utility 'classify_articles.py'. We used the following command to classify articles using the pretrained classifier model (calling to the parsed and extracted article pickle files written to the 'results/ents' directory - see above):

In the terminal:
```
python journal_classification/classify_articles.py -i results/ents/original -o results -m journal_classification/nb_classifier.pickle  
```
This step writes out a journal-discipline dictionary to a pickle file 'article_domain_prediction.pickle' in the output directory for further use.

# Analytic Method Entity Preprocessing
The analytic method entities extracted from the NER algorithm are represented as a sequence of characters formatted as strings. The final step of our preprocessing pipeline consisted of a sequence of preprocessing algorithms designed to clean the analytic method entity strings for further analysis. These steps include lower-casing, removal of non-alphanumeric characters, lemmatization, and spell-checking using the symspell package (https://github.com/wolfgarbe/SymSpell). The python scripts for this step are provided in the 'entity_preprocessing' directory. To run this preprocessing step we provide a command-line utility - 'preprocess_entities.py'. This utility calls to the 'preprocessing.py' script that defines a 'EntityPreprocessing' class containing all methods used for preprocessing. Within the sub-directory 'results/ents', we created a sub-directory 'preprocessed' to write preprocessed articles from this step. To preprocess entity strings we used the following command (calling to the parsed and extracted article pickle files written to the 'results/ents/orig' directory - see above):

In the terminal:
```
python entity_preprocessing/preprocess_entities.py -i results/ents/original -o results/ents/preprocesing -m results/article_domain_prediction.pickle 
```











```

```
