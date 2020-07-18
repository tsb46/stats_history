import re
import pandas as pd
import pickle
import string
import scipy.sparse as sp

from analysis_main.utils import flatten_nested_list
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from symspellpy import SymSpell, Verbosity
from journal_classification.custom_transformers import AbstractSelector, \
    JournalTitleSelector, TitleSelector

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()


class EntityPreprocessing:
    def __init__(self, domain_classifier, classifier_type, n_thres=5,
                 ignore_article_counts=True, precomp_ent_groups=None,
                 n_spell_check_thres=5):
        if classifier_type == 'alg':
            self.domain_classifier = pickle.load(open(domain_classifier, 'rb'))
            self.classifier_type = 'alg'
        elif classifier_type == 'precomputed':
            self.domain_classifier = domain_classifier
            self.classifier_type = 'precomputed'
        self.n_thres = n_thres
        self.ignore_article_counts = ignore_article_counts
        self.precomp_ent_groups = precomp_ent_groups
        self.n_spell_check_thres = n_spell_check_thres

    def preprocess(self, articles_dict):
        # convert pubmed article dicts to list of named entities per article
        ents_list = [articles_dict[pmid]['ents'] for pmid in articles_dict.keys()]
        ents_orig = list(set(flatten_nested_list(ents_list)))
        # Preprocess entity strings - removal of bad chars, lemmatize, etc.
        ents_proc = [self._preprocess_string(ent_string)
                     for ent_string in ents_orig]
        # Replace entities per article with PREPROCESSED entity strings and create Counter
        ents_processing_history = self._create_process_history(ents_orig, ents_proc)
        ents_list_proc = self._replace_ents(
            ents_list, ents_processing_history, 'processed'
        )
        ents_counter = Counter(flatten_nested_list(ents_list_proc))
        # Spell check preprocessed entity strings
        spell_checker = self._create_spell_checker_dict(ents_counter,
                                                        self.n_spell_check_thres)
        ents_corrected = [self._spell_check(ent_string, spell_checker) 
                          if ents_counter[ent_string] < self.n_spell_check_thres
                          else ent_string
                          for ent_string in ents_proc]
        # Replace entities per article with CORRECTED entity strings and create Counter
        ents_processing_history = self._create_process_history(ents_orig, ents_proc, ents_corrected)
        ents_list_corrected = self._replace_ents(
            ents_list, ents_processing_history, 'corrected'
        )
        # If precomputed groupings of entities are supplied (as a dict) then replace
        # entity labels with grouping labels
        if self.precomp_ent_groups is not None:
            ents_list_corrected = self._group_ents(ents_list_corrected, 
                                                   self.precomp_ent_groups)
        # Count up entity occurrences
        if self.ignore_article_counts:
            ents_counter = Counter([ent.lower() for article in ents_list_corrected
                                    for ent in list(set(article))])
        else:
            ents_counter = Counter(flatten_nested_list(ents_list_corrected))
        # Threshold entities by their counts - specified by user: 'n_thres'
        ents_vocab_count = {ent: count for ent, count in ents_counter.items()
                            if count > self.n_thres}
        # Replace entities per article with final entity dictionary
        ents_final = self._package_final_ent_dict(ents_list_corrected,
                                                  ents_vocab_count,
                                                  articles_dict)
        # Free up memory
        del articles_dict, ents_list, ents_list_proc, \
            ents_counter, ents_orig, ents_proc
        # Get Final entity vocab counts
        final_ent_vocab_count = Counter(
            flatten_nested_list([ents_final[pmid]['ents'] for pmid in ents_final])
        )
        # Compute word count matrix
        word_count_mat, voc2id, pmid2id = self._compute_word_count_mat(
            ents_final, ents_vocab_count)
        # Classify articles according to their scientific domain
        ents_final = self._classify_articles(ents_final,
                                             self.domain_classifier, 
                                             self.classifier_type)
        final_dict = {
            'word_count_matrix':
                {
                'matrix': word_count_mat,
                'voc2id': voc2id,
                'pmid2id': pmid2id                },
            'processing_history': ents_processing_history,
            'vocab_count': final_ent_vocab_count,
            'ents': ents_final
        }
        return final_dict

    @staticmethod
    def _classify_articles(ents_final, nb_classifier, classifier_type):
        if classifier_type == 'alg':
            articles_data = []
            for pmid in ents_final:
                articles_data.append(
                    {
                        'abstract': ents_final[pmid]['abstract'],
                        'title': ents_final[pmid]['title'],
                        'journal': ents_final[pmid]['journal'],
                        'pmid': pmid
                     }
                )
            article_df = pd.DataFrame(articles_data)
            predicted_domains = nb_classifier.predict(article_df)
            for pmid, pred_domain in zip(ents_final, predicted_domains):
                ents_final[pmid].update({'domain': pred_domain})
                # Remove abstract - too much memory
                del ents_final[pmid]['abstract']
        elif classifier_type == 'precomputed':
            for pmid, pred_domain in nb_classifier.items():
                if pmid in ents_final:
                    ents_final[pmid].update({'domain': pred_domain})

        return ents_final
    
    @staticmethod
    def _create_process_history(ents_orig, ents_proc, ents_corrected=None):
        if ents_corrected is None:
            ents_processing_history = {orig:
                {
                    'processed': proc
                }
                for orig, proc
                in zip(ents_orig, ents_proc)}
        else:
            ents_processing_history = {orig:
                {
                    'processed': proc,
                    'corrected': correct
                }
                for orig, proc, correct
                in zip(ents_orig, ents_proc, ents_corrected)}
        return ents_processing_history
        
    @staticmethod
    def _create_spell_checker_dict(ents_counter, thres):
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for ent, count in ents_counter.items():
            if count > thres:
                sym_spell.create_dictionary_entry(ent, count)
        return sym_spell

    def _compute_word_count_mat(self, ents_final, ents_dict):
        voc2id = dict(zip(ents_dict.keys(), range(len(ents_dict))))
        pmid_list = [pmid for pmid in ents_final]
        ents_list = [article['ents'] for article in ents_final.values()]
        rows, cols, vals = [], [], []
        for r, d in enumerate(ents_list):
            for e in d:
                if voc2id.get(e) is not None:
                    rows.append(r)
                    cols.append(voc2id[e])
                    vals.append(1)
        word_count_mat = sp.csr_matrix((vals, (rows, cols)))
        # If the user specifies that we ignore entity counts within article: binarize
        if self.ignore_article_counts:
            word_count_mat[word_count_mat > 0] = 1

        return word_count_mat, voc2id, pmid_list
    
    @staticmethod
    def _group_ents(ent_list, ent_groups):
        replaced_ent_list = []
        for article_ents in ent_list:
            if article_ents is not None:
                new_article_ents = [ent_groups[ent]
                                    for ent in article_ents 
                                    if ent in ent_groups]
            else:
                new_article_ents = []
            replaced_ent_list.append(new_article_ents)
        return replaced_ent_list
    
    @staticmethod
    def _package_final_ent_dict(ent_list, ent_vocab_dict, articles_dict):
        pmid_list = [pmid for pmid in articles_dict.keys()]
        final_ent_dict = {}
        for article_ents, pmid in zip(ent_list, pmid_list):
            if len(article_ents) > 0:
                new_article_ents = [ent for ent in article_ents
                                    if ent in ent_vocab_dict]
                final_ent_dict.update(
                    {
                        pmid: {
                            'ents': new_article_ents,
                            'date': articles_dict[pmid]['date'],
                            'citations': articles_dict[pmid]['citations'],
                            'journal': articles_dict[pmid]['journal'],
                            'title': articles_dict[pmid]['title'],
                            'abstract': articles_dict[pmid]['abstract']
                        }
                    }
                )
        return final_ent_dict
    
    @staticmethod
    def _preprocess_string(string_x):
        # replace '-' with white space
        string_x = re.sub(r'[-‐−]', ' ', string_x)
        # remove digits
        string_nodigits = re.sub(r'[0-9]+', '', string_x)
        # remove apostrophes
        string_noapos = re.sub(r"['’]", '', string_nodigits)
        # tokenize string
        string_tokenized = word_tokenize(string_noapos)
        # remove punctuation
        string_nopunc = [''.join([char for char in token
                                  if char not in string.punctuation])
                         for token in string_tokenized]
        # lower-case and lemmatize
        string_lemm = [lemmatizer.lemmatize(token.lower())
                       for token in string_nopunc]
        # Remove whitespace on beginning or end
        string_stripped = ' '.join(string_lemm).strip().replace('  ', ' ')
        return string_stripped

    @staticmethod
    def _replace_ents(ent_list, ent_vocab_dict, field):
        replaced_ent_list = []
        for article_ents in ent_list:
            if article_ents is not None:
                new_article_ents = [ent_vocab_dict[ent][field]
                                    for ent in article_ents]
            else:
                new_article_ents = []
            replaced_ent_list.append(new_article_ents)
        return replaced_ent_list

    @staticmethod
    def _spell_check(string_x, spell_checker):
        spell_correction = spell_checker.lookup(string_x,
                                                Verbosity.CLOSEST,
                                                max_edit_distance=2,
                                                include_unknown=True)
        corrected_string = spell_correction[0].term
        return corrected_string


