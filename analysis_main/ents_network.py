import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis_main.ents_base import EntityBase
from datetime import datetime
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import squareform
import tensortools as tt


class EntityNetwork(EntityBase):
    def __init__(self, ents_fp, ent_grouper=None, year_range=None, 
                 ignore_article_counts=True):
        super().__init__(ents_fp, ent_grouper, year_range, ignore_article_counts)
    
    def compute_network(self, word_count_mat):
        network = self._compute_sim_mat(word_count_mat, norm=True)
        return network
    
    def compute_network_by_domain(self, mat, domains):
        network_by_domain = []
        network_domains = domains.unique()
        for domain in network_domains:
            domain_indices = np.where(domains == domain)[0]
            network = self._compute_sim_mat(mat[domain_indices, :], norm=True)
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
        markerline, stemlines, baseline = plt.stem(self.network_domains, self.factors[2][:,comp], ':')
        plt.setp(stemlines, 'linewidth', 3)
        plt.xticks(rotation=90, size=15, weight='bold')
        plt.xlabel('Domain')
        plt.ylabel('Component Weight')
        plt.title(f'Tensor Component {comp}')
            
    def tensor_decomp(self, network_tensor, n_comp):
        self.ent_vocab = [self.id2voc[i] for i in range(len(self.id2voc))]
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
    def _norm_ppmi_mat(nppmi_mat):
        nppmi_vec = squareform(nppmi_mat.toarray(), 'tovector')
        norm_nppmi_vec = Normalizer('l2').fit_transform(nppmi_vec.reshape(1,-1))
        norm_nppmi_mat = squareform(norm_nppmi_vec.reshape(-1,), 'tomatrix')
        return norm_nppmi_mat
    
    
    
        