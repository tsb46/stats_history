import re
import pandas as pd
import pickle
import string

from utils import flatten_nested_list
from collections import Counter
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from symspellpy import SymSpell, Verbosity

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()


class EntityPreprocessing:
    def __init__(self, domain_dict, n_thres=10,
                 ignore_article_counts=True, n_spell_check_thres=5):
        """
        params:
        domain_dict: dictionary containing article pmcid (key) to predicted domain for article (value)
        n_thres: # of number of instances across corpus an entity must have to be included in final entity list
        ignore_article_counts: ignore the number of instances of an entity within an article
        ent_categories: dictionary containing a categorization of entities into pre-specified categories
        n_spell_check_thres: any entity with a number of instances across the corpus below this number is 
            spell-checked
        """
        self.domain_classifier = domain_dict
        self.n_thres = n_thres
        self.ignore_article_counts = ignore_article_counts
        self.n_spell_check_thres = n_spell_check_thres


    def preprocess_articles(self, article_dict):
        article_proc_dict = {}
        for article in article_dict:
            if article is not None:
                if article['parsed'] and len(article['ents']) > 0:
                    # Preprocess entity strings - removal of bad chars, lemmatize, etc.
                    ents_proc = [self._preprocess_string(ent_string) for ent_string in article['ents']]
                    # Get predicted domain for journal
                    article_domain = self.domain_classifier[article['pmcid']]
                    article_proc_dict[article['pmcid']] = {
                                                          'journal': article['journal'],
                                                          'domain': article_domain,
                                                          'year': article['year'],
                                                          'date': article['date'],
                                                          'ents': article['ents'],
                                                          'ents_proc1': ents_proc
                                                          }
        return article_proc_dict


    def preprocess_corpus(self, ents_corpus):
        ents_corpus_list = [ents_corpus[pmcid]['ents_proc1'] for pmcid in ents_corpus]
        ents_corpus_list_flat = flatten_nested_list(ents_corpus_list)
        # Get unique entity strings for pre-processing
        unique_ents = list(set(ents_corpus_list_flat))
        ents_counter = Counter(ents_corpus_list_flat)
        # Spell check preprocessed entity strings
        spell_checker = self._create_spell_checker_dict(ents_counter,
                                                        self.n_spell_check_thres)
        unique_ents_proc = [self._spell_check(ent_string, spell_checker) 
                            if ents_counter[ent_string] < self.n_spell_check_thres
                            else ent_string
                            for ent_string in unique_ents]
        # Replace entities per article with spell corrected entity strings and create Counter
        orig_to_proc = {orig: proc for orig, proc in zip(unique_ents, unique_ents_proc)}
        # proc_to_orig = {proc: orig for orig, proc in zip(unique_ents, unique_ents_proc)}
        ents_corpus_list_proc = self._replace_ents(ents_corpus_list, orig_to_proc)
        # Count up entity occurrences
        if self.ignore_article_counts:
            ents_counter = Counter([ent.lower() for article in ents_corpus_list_proc
                                    for ent in list(set(article))])
        else:
            ents_counter = Counter(flatten_nested_list(ents_corpus_list_proc))
        # Threshold entities by their counts - specified by user: 'n_thres'
        proc_to_final = {ent: (ent if count > self.n_thres else None) 
                         for ent, count in ents_counter.items()}
        # Replace entities per article with final entity dictionary
        ents_corpus_final = self._update_article_dict(ents_corpus,
                                                      orig_to_proc,
                                                      proc_to_final)

        return ents_corpus_final
     
    @staticmethod
    def _create_spell_checker_dict(ents_counter, thres):
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for ent, count in ents_counter.items():
            if count > thres:
                sym_spell.create_dictionary_entry(ent, count)
        return sym_spell
    
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
    def _replace_ents(ent_list, orig_to_proc):
        replaced_ent_list = []
        for article_ents in ent_list:
            new_article_ents = [orig_to_proc[ent] for ent in article_ents]
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

    @staticmethod
    def _update_article_dict(ents_corpus, orig_to_proc, proc_to_final):
        for pmcid in ents_corpus:
            ents_corpus[pmcid].update({
                                       'ents_proc2': [orig_to_proc[ent] 
                                                      for ent in ents_corpus[pmcid]['ents_proc1']],
                                       'ents_final': [proc_to_final[orig_to_proc[ent]] 
                                                      for ent in ents_corpus[pmcid]['ents_proc1']]             
                                       })
        return ents_corpus


