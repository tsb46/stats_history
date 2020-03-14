import argparse
import json
import os
import pandas as pd
import pickle

from glob import glob
# from sklearn.base import BaseEstimator, TransformerMixin

#
# class AbstractSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, field):
#         self.field = field
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X[self.field]
#
#
# class JournalTitleSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, field):
#         self.field = field
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X[self.field]
#
#
# class TitleSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, field):
#         self.field = field
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return X[self.field]


# def classify_article(article, metadata, nb_classifier):
#     try:
#         import pdb; pdb.set_trace()
#         article_data = {
#             'journal': metadata[article['pmid']]['journal'],
#             'abstract': article['abstract'],
#             'title': article['title']
#         }
#         article_series = pd.DataFrame(article_data, index=0)
#         domain = nb_classifier.predict(article_series)
#         return domain
#     # If any crucial article info is missing, return None
#     except KeyError:
#         return None


def pull_ents(file_path, ner_model, output_dir, article_metadict):
    print(file_path)
    ents_list = []
    articles = json.load(open(file_path, 'r'))
    for article in articles:
        metadata = pull_metadata(article['pmid'], article_metadict)
        if metadata is not None:
            # domain = classify_article(article, article_metadict, classifier)
            # if domain is not None:
            try:
                spacy_obj = ner_model(article['methods'])
                ents = [str(ent) for ent in spacy_obj.ents]
            except AttributeError:
                ents = []
            ents_dict = {
                'title': article['title'],
                'abstract': article['abstract'],
                'pmid': article['pmid'],
                'date': article['date'],
                'journal': metadata['journal'],
                'citations': metadata['citations'],
                'ents': ents
            }
            ents_list.append(ents_dict)
    file_out_ext = os.path.basename(file_path)
    file_out = os.path.splitext(file_out_ext)[0]
    pickle.dump(ents_list, open(output_dir + '/ents/' + file_out + '.pickle', 'wb'))


def pull_metadata(pmid, article_metadict):
    if pmid in article_metadict:
        journal = article_metadict[pmid]['journal']
        citation = article_metadict[pmid]['citations']
        return {'journal': journal, 'citations': citation}
    else:
        return None


def run_main(input_dir, output_dir, spacy_model, article_metadict):
    # Get paths to necessary inputs
    corpus_files = glob(input_dir + '/preprocessed/*.json')
    ner_model = pickle.load(open(spacy_model, 'rb'))
    # nb_classifier = pickle.load(open(journal_classifier, 'rb'))
    metadict = json.load(open(article_metadict, 'rb'))
    # Extract entities from pubmed corpus files
    for file_path in corpus_files:
        pull_ents(file_path, ner_model, output_dir, metadict)


if __name__ == '__main__':
    """Extract named entities from pubmed corpus"""
    parser = argparse.ArgumentParser(description='Extract named entities from '
                                                 'pubmed corpus files')
    parser.add_argument('-d', '--corpus_dir',
                        help='<Required> Path to directory containing '
                             'pubmed corpus files',
                        required=True,
                        type=str)
    parser.add_argument('-o', '--output_dir',
                        help='<Required> Path to directory '
                             'to write outputs (.pickle files) to',
                        required=True,
                        type=str)
    parser.add_argument('-s', '--spacy_ner_model',
                        help='<Required> path to trained spacy ner '
                             'model .pickle file',
                        required=True,
                        type=str)
    # parser.add_argument('-c', '--journal_classifier',
    #                     help='<Required> path to trained nb classifier to '
    #                          'categorize articles',
    #                     required=True,
    #                     type=str)
    parser.add_argument('-a', '--article_metadict',
                        help='<Required> path to article metadict for pulling'
                             'relevant metainfo',
                        required=True,
                        type=str)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['corpus_dir'], args_dict['output_dir'],
             args_dict['spacy_ner_model'],
             args_dict['article_metadict'])