import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp


from glob import glob
from collections import Counter
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

np.random.seed(0)

class EntityBase:
    def __init__(self, ents_fp, ent_grouper=None, year_range=None, 
                 ignore_article_counts=True):
        self.ent_file = ents_fp
        if ent_grouper is not None:
          self.group_ents = True
        else:
          self.group_ents = False
        self.ent_grouper = ent_grouper
        self.ignore_article_counts = ignore_article_counts
        if year_range is not None:
          self.year_range = year_range
        else:
          self.year_range = (float('-inf'), float('inf'))
    
    def convert_to_dataframe(self, additional_cols=None):
        """
        Convert word count matrix to dataframe for further processing, index is set as pmcid
        :param additional_cols: a list containing strings referring to additional article info.
         These are included included as extra columns in the dataframe. Options include: 'date',
         'journal', 'title', 'citations' and/or 'domain'.
        :return: word count dataframe
        """
        article_dict = self._pull_ents(self.ent_file)
        if self.group_ents:
          ent_grouper = self._create_ent_grouper(self.ent_grouper)
          self.ent_group = ent_grouper
        else:
          ent_grouper = None
        word_count_mat, voc2id, pmcid_all = self._compute_word_count_mat(article_dict, ent_grouper, 
                                                                         self.year_range)
        id2voc = {indx: ent for ent, indx in voc2id.items()}
        self.voc2id = voc2id
        self.id2voc = id2voc
        word_count_df = pd.DataFrame(
            word_count_mat.todense(),
            index=pmcid_all,
            columns=[id2voc[indx] for indx in range(len(id2voc))]
        )
        # If the user specified extra columns of article info to be added
        if additional_cols is not None:
            additional_data = {pmcid: {col: article_dict[pmcid][col] for col in additional_cols} 
                               for pmcid in pmcid_all}
            additional_data_df = pd.DataFrame.from_dict(additional_data, 
                                                        orient='index')
            if 'year' in additional_cols:
              additional_data_df['year'] = additional_data_df['year'].astype(int)
            # additional_data_df = additional_data_df.reindex(word_count_df.index)
        else:

            additional_data_df = [] 

        return word_count_df, additional_data_df

    @staticmethod
    def nmf_decompose(word_count_df, n_components, norm=True):
      nmf = NMF(n_components=n_components, init='random', random_state=0)
      if norm:
        word_count_df = normalize(word_count_df, norm='l2', axis=0)
      nmf.fit(word_count_df)
      return nmf

    def _compute_word_count_mat(self, article_dict, ent_grouper, year_range):
        pmcid_all = [pmcid for pmcid in article_dict 
                     if year_range[0] <= int(article_dict[pmcid]['year']) <= year_range[1]]
        ents_list = [article_dict[pmcid]['ents_final'] for pmcid in pmcid_all]
        unique_ents = sorted(list(set([ent for article_ents in ents_list 
                                for ent in article_ents
                                if ent is not None])))
        if ent_grouper is None:
          ent_grouper = {orig_ent: orig_ent for orig_ent in unique_ents}
        else:
          unique_ents = sorted(list(set([ent_grouper[ent] for ent in unique_ents 
                                  if ent_grouper.get(ent) is not None])))
        voc2id = dict(zip(unique_ents, range(len(unique_ents))))
        rows, cols, vals = [], [], []
        for r, d in enumerate(ents_list):
          for e in d:
              if voc2id.get(ent_grouper.get(e)) is not None:
                  rows.append(r)
                  cols.append(voc2id[ent_grouper[e]])
                  vals.append(1)
        word_count_mat = sp.csr_matrix((vals, (rows, cols)),
                                       shape=(len(pmcid_all), len(unique_ents)))
        # If the user specifies that we ignore entity counts within article: binarize
        if self.ignore_article_counts:
            word_count_mat[word_count_mat > 1] = 1

        return word_count_mat, voc2id, pmcid_all

    @staticmethod
    def _create_ent_grouper(ent_grouper):
      # drop ents with no classification
      ent_grouper = {ent: group for ent, group in ent_grouper.items() 
                     if group is not None
                     if ~pd.isnull(group)}
      return ent_grouper

    @staticmethod
    def _pull_ents(ent_file):
      article_dict = pickle.load(open(ent_file, 'rb'))               
      return article_dict
