import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_main.ents_base import EntityBase
from scipy.stats import zscore
from sklearn.preprocessing import normalize 
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


class EntityDomain(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, precomp_ent_group=None):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts, precomp_ent_group)
        # Hard-coded metadata columns in the word count dataframe - intentionally 
        # dropped in some analyses
        self.metadata_cols = ['citations', 'journal', 
                              'title', 'abstract', 'date',
                              'article_counts']
    @staticmethod
    def comp_univariate_means(lda_dict, comp):
        labels = lda_dict['label_encoder']
        domains = lda_dict['domain_coded']
        scores = lda_dict['scores']
        univariate_means = [(d, scores[domains==i, comp].mean()) 
                            for i, d in enumerate(labels.classes_)]
        return univariate_means

    @staticmethod
    def get_top_ents(classifier_obj, labels, comp, top_n=10, neg=True):
        comp_coef = classifier_obj.coef_[comp,:]
        comp_coef_norm = zscore(comp_coef)
        top_n_indx = np.argsort(comp_coef_norm)[-top_n:][::-1]
        print ('Rank  Label  Coef')
        for rank, n in enumerate(top_n_indx):
            print(f'{rank+1} - {labels[n]}: {comp_coef_norm[n]}')
        if neg:
            top_n_indx = np.argsort(comp_coef_norm)[:top_n]
            print ('Rank  Label  Coef')
            for rank, n in enumerate(top_n_indx):
                print(f'{rank+1} - {labels[n]}: {comp_coef_norm[n]}')
    
    def ent_counts_per_domain(self, word_count_df, normalize=False):
        word_counts = word_count_df.drop(columns=self.metadata_cols, errors='ignore')
        counts_per_domain = word_counts.groupby('domain').agg('sum')
        if normalize:
            counts_per_domain = counts_per_domain.apply(lambda x: x/ x.sum(), axis=1)
        return counts_per_domain.T
    
    @staticmethod
    def plot_lda_scores(lda_dict, comps=(1,2), n_samples_per_domain=1000):
        labels = lda_dict['label_encoder']
        domains = lda_dict['domain_coded']
        scores = lda_dict['scores']
        n = n_samples_per_domain
        plt.figure(figsize=(20,20))
        for indx, target_name in enumerate(labels.classes_):
            x = scores[domains == indx, comps[0]]
            y = scores[domains == indx, comps[1]]
            rand_perm = np.random.permutation(x.shape[0])
            plt.scatter(x[rand_perm[:n]], y[rand_perm[:n]], alpha=.5,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.show()
    
    def run_lda(self, word_count_matrix, domains, n_dim):
        le = LabelEncoder()
        domain_coded = le.fit_transform(domains)
        lda = LinearDiscriminantAnalysis(n_components=n_dim, solver='eigen')
        word_count_matrix = word_count_matrix.todense()
        word_count_matrix = StandardScaler().fit_transform(word_count_matrix)
        lda.fit(word_count_matrix, domain_coded)
        lda_scores = lda.transform(word_count_matrix)
        self.lda = {
            'lda_obj': lda,
            'scores': lda_scores,
            'label_encoder': le,
            'domain_coded': domain_coded
        }
        return self.lda
    
    def classify_domains(self, word_count_matrix, domains):
        le = LabelEncoder()
        domain_coded = le.fit_transform(domains)
        word_count_scaled = StandardScaler(with_mean=False).fit_transform(word_count_matrix)
        logit = LogisticRegression(class_weight='balanced', solver='sag', multi_class='ovr')
        logit.fit(word_count_scaled, domain_coded)
        self.logit = {
            'logit_obj': logit,
            'label_encoder': le,
            'domain_coded': domain_coded
        }
        return self.logit
            

        