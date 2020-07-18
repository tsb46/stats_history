import pickle
import pandas as pd
import numpy as np

from analysis_main.preprocessing import EntityPreprocessing
from glob import glob
from collections import Counter
from sklearn.decomposition import NMF, SparsePCA, DictionaryLearning
from sklearn.preprocessing import normalize 


class EntityBase:
    def __init__(self, ents_dir, journal_classifier, ent_count_thres=10,
                 ignore_article_counts=True, precomp_ent_group=None):
        ent_files = glob(ents_dir + '/*.pickle')
        self.ent_files = ent_files
        if isinstance(journal_classifier, str):
            self.classification_type = 'alg'
        elif isinstance(journal_classifier, dict):
            self.classification_type = 'precomputed'
        else:
            raise Exception('classifier input must be path to spacy classifier '
                            'model or dict containing pre-computed classifications')
        self.classifier = journal_classifier
        self.ent_count_thres = ent_count_thres
        self.ignore_article_counts = ignore_article_counts
        self.precomp_ent_group = precomp_ent_group
            
    def convert_to_dataframe(self, additional_cols=None):
        """
        Convert word count matrix to dataframe for further processing, index is set as pmid
        :param additional_cols: a list containing strings referring to additional article info.
         These are included included as extra columns in the dataframe. Options include: 'date',
         'journal', 'title', 'citations' and/or 'domain'.
        :return: word count dataframe
        """
        id2voc = {indx: ent for ent, indx in
                  self.ents['word_count_matrix']['voc2id'].items()}
        word_count_df = pd.DataFrame(
            self.ents['word_count_matrix']['matrix'].todense(),
            index=self.ents['word_count_matrix']['pmid2id'],
            columns=[id2voc[indx] for indx in range(len(id2voc))]
        )
        # If the user specified extra columns of article info be added
        if additional_cols is not None:
            additional_data = {pmid: {col: self.ents['ents'][pmid][col] 
                                      for col in additional_cols} 
                              for pmid in self.ents['ents']}
            additional_data_df = pd.DataFrame.from_dict(additional_data, 
                                                        orient='index')
            additional_data_df = additional_data_df.reindex(word_count_df.index)
            return word_count_df, additional_data_df

        return word_count_df
    
    def ent_counts_per_article(self):
        ents_len = [len(entry['ents']) for entry in self.article_dicts]
        ents_len_counter = Counter(ents_len)
        return ents_len_counter
    
    def process_ents(self):
        article_dicts = self._pull_ents()
        # convert list of pubmed dicts to dictionary indexed by pmid
        article_dicts = {article['pmid']: article for article in article_dicts}
        ents_preprocess = EntityPreprocessing(self.classifier, self.classification_type,
                                              self.ent_count_thres,
                                              self.ignore_article_counts, 
                                              self.precomp_ent_group)
        self.ents = ents_preprocess.preprocess(article_dicts)
        
    def _pull_ents(self):
        article_dicts = []
        for file in self.ent_files:
            ents = pickle.load(open(file, 'rb'))
            article_dicts += ents
        return article_dicts
