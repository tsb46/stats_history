import argparse
import numpy as np
import pandas as pd
import pickle

from journal_classification.custom_transformers import AbstractSelector, \
    JournalTitleSelector, TitleSelector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedKFold


def create_nb_classifier_pipeline(n_features):
    classifier = Pipeline([
        ('features', FeatureUnion([
            ('journal_title', Pipeline([
                ('colext', JournalTitleSelector('journal')),
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3),
                                          min_df=0.0005,
                                          max_df=0.6,
                                          strip_accents='ascii')),
            ])),
            ('article_title', Pipeline([
                ('colext', TitleSelector('title')),
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3),
                                          min_df=0.001,
                                          max_df=0.6,
                                          strip_accents='ascii',
                                          sublinear_tf=True))
            ])),
            ('article_abstract', Pipeline([
                ('colext', AbstractSelector('abstract')),
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3),
                                          min_df=0.001,
                                          max_df=0.6,
                                          strip_accents='ascii',
                                          sublinear_tf=True))
            ])),
        ])),
        ('feature_selection', SelectKBest(chi2, k=n_features)),
        ('clf', ComplementNB())
    ])
    return classifier


def run_classifier_cv(training_data, nb_classifier, n_folds, n_repeats):
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)
    y = training_data['classification']
    X = training_data[['title', 'abstract', 'journal']]
    all_scores = []
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        nb_classifier.fit(X_train, y_train)
        preds = nb_classifier.predict(X_test)
        score = accuracy_score(y_test, preds)
        all_scores.append(score)
    print(f'Total accuracy: %{np.mean(all_scores)}')


def train_final_classifier(training_data, nb_classifier):
    y = training_data['classification']
    X = training_data[['title', 'abstract', 'journal']]
    nb_classifier.fit(X, y)
    return nb_classifier


def run_main(training_samples, output_dir, n_folds, n_repeats, n_features):
    # Get paths to necessary inputs
    training_data = pd.read_csv(training_samples)
    # Drop any training data missing abstract
    training_data = training_data.dropna(subset=['abstract'])
    # Create sklearn pipeline for multinomial NB classifier
    nb_classifier = create_nb_classifier_pipeline(n_features)
    # Run classifier cross-validation across n-repeated K folds
    run_classifier_cv(training_data, nb_classifier, n_folds, n_repeats)
    # Fit final classifier model to all training data
    nb_classifier_fitted = train_final_classifier(training_data, nb_classifier)
    # Pickle final model to final directory
    pickle.dump(nb_classifier_fitted,
                open(output_dir + '/nb_classifier.pickle', 'wb'))


if __name__ == '__main__':
    """Train (complementary) multinomial NB algorithm to classify
    scientific articles into their respective domains"""

    parser = argparse.ArgumentParser(description='Train naive bayes classifier')
    parser.add_argument('-i', '--input_training',
                        help='<Required> Path to .csv file containing labeled '
                             'training samples',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write trained model to',
                        required=True,
                        type=str)
    parser.add_argument('-nf', '--n_folds',
                        help='<Required> # of folds in repeated '
                             'K-fold cross-validation',
                        required=False,
                        default=5,
                        type=str)
    parser.add_argument('-nr', '--n_repeats',
                        help='<Required> # of repeats in repeated '
                             'K-fold cross-validation',
                        required=False,
                        default=5,
                        type=str)
    parser.add_argument('-nfeat', '--n_features',
                        help='<Required> # of features selected from'
                             'chi-squared feature selection algorithm',
                        required=False,
                        default=15000,
                        type=str)
    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_training'], args_dict['output_dir'],
             int(args_dict['n_folds']), int(args_dict['n_repeats']),
             int(args_dict['n_features']))
