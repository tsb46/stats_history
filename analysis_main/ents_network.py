import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis_main.ents_base import EntityBase
from datetime import datetime
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import squareform
from tensorly.decomposition import parafac
import tensortools as tt


class EntityNetwork(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, precomp_ent_group=None):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts, precomp_ent_group)
        # Hard-coded metadata columns in the word count dataframe - intentionally 
        # dropped in some analyses
        self.metadata_cols = ['citations', 'journal', 
                              'title', 'abstract', 'domain']
    
    def compute_network(self, word_count_mat):
        network = self._compute_sim_mat(word_count_mat, norm=True)
        return network
            
    def compute_network_by_time(self, word_count_mat, article_date, time_res='annual', start_year = 2009, remove_most_recent=False):
        date_resample = self._downsample_time(article_date, time_res)
        date_resample = self._threshold_dates(date_resample, start_year, remove_most_recent)
        network_by_time = []
        network_dates = date_resample.sort_values().unique()
        for date in network_dates:
                date_indices = np.where(date_resample == date)[0]
                network = self._compute_sim_mat(word_count_mat[date_indices, :], norm=True)
                network_by_time.append(network)
        network_by_time_array = np.dstack(network_by_time)
        self.network_dates = network_dates
        return network_by_time_array
    
    def compute_network_by_domain(self, word_count_mat, article_domains):
        network_by_domain = []
        network_domains = article_domains.unique()
        for domain in network_domains:
            domain_indices = np.where(article_domains == domain)[0]
            network = self._compute_sim_mat(word_count_mat[domain_indices, :], norm=True)
            network_by_domain.append(network)
        network_by_domain_array = np.dstack(network_by_domain)
        self.network_domains = network_domains
        return network_by_domain_array
    
    def tensor_decomp_query(self, comp, display_opt, top_n=10):
        weight_vec = self.factors[0][:,comp]
        sort_indices = np.flip(weight_vec.argsort())
        for i in range(top_n):
            print('{} : {} \n'.format(self.ent_vocab[sort_indices[i]],
                                      weight_vec[sort_indices[i]]))
        if display_opt == 'time':
            plt.plot(self.factors[2][:,comp])
            plt.xticks(range(factors[2].shape[1]), self.network_dates)
            plt.xlabel('Time')
            plt.ylabel('Component Weight')
            plt.title(f'Tensor Component {comp}')
        elif display_opt == 'domain':
            plt.stem(self.network_domains, self.factors[2][:,comp])
            plt.xticks(rotation=45)
            plt.xlabel('Domain')
        plt.ylabel('Component Weight')
        plt.title(f'Tensor Component {comp}')
            
    def tensor_decomp(self, network_tensor, n_comp, voc2id):
        id2voc = {id: voc for voc, id in voc2id.items()}
        self.ent_vocab = [id2voc[i] for i in range(len(voc2id))]
#         self.weights, self.factors = parafac(network_tensor, n_comp, init='random', 
#                                              non_negative=True, normalize_factors=True)
        tensors = tt.ncp_bcd(network_tensor, n_comp, verbose=False)
        self.factors = tensors.factors.rebalance()
    
    def _compute_sim_mat(self, word_count_mat, norm):
        # Compute word co-occurence matrix
        word_cc = word_count_mat.T * word_count_mat
        # Set diagonal to zero
        word_cc = sp.lil_matrix(word_cc)
        word_cc.setdiag(0)
        word_cc = sp.csr_matrix(word_cc)
        word_cc.data = np.log(word_cc.data)
        sim_mat = word_cc
        # Compute PMI Matrix
#         sim_mat = self._convert_to_ppmi_mat(word_cc)
        if norm:
            sim_mat = self._norm_ppmi_mat(sim_mat)
        else:
            sim_mat = sim_mat.toarray()
        return sim_mat
    
    @staticmethod
    def _convert_to_ppmi_mat(word_cc):
        # total word counts
        Z = word_cc.sum()

        # counts per article.
        Zr = np.array(word_cc.sum(axis=1), dtype=np.float64).flatten()

        # Get indices of non zero elements
        ii, jj = word_cc.nonzero()  # row, column indices
        Cij = np.array(word_cc[ii,jj]).flatten()

        # calc positive PMI
        pmi = np.log( (Cij * Z) / (Zr[ii] * Zr[jj]))
        ppmi = np.maximum(0, pmi)  # take positive only
        # Create sparse matrix
        ppmi_mat = sp.csc_matrix((ppmi, (ii,jj)), shape=word_cc.shape,
                                      dtype=np.float64)
        ppmi_mat.eliminate_zeros()  

        return ppmi_mat
        
    @staticmethod
    def _downsample_time(series, time_res):
        date_series = pd.to_datetime(series.values, 
                                     format='%d-%m-%Y',
                                     errors='coerce')
        if time_res == 'annual':
            resampled_series = date_series.year
        else:
            raise Exception('Provided time resolution not available')
        return resampled_series
    
    @staticmethod
    def _norm_ppmi_mat(nppmi_mat):
        nppmi_vec = squareform(nppmi_mat.toarray(), 'tovector')
        norm_nppmi_vec = Normalizer('l2').fit_transform(nppmi_vec.reshape(1,-1))
        norm_nppmi_mat = squareform(norm_nppmi_vec.reshape(-1,), 'tomatrix')
        return norm_nppmi_mat
    
    @staticmethod
    def _threshold_dates(date_series, start_year, remove_most_recent=False):
        time_mask = date_series >= start_year
        thres_date_series = date_series[time_mask]
        if remove_most_recent:
            thres_date_series = thres_date_series[:-1].copy()
        return thres_date_series
    
    
        