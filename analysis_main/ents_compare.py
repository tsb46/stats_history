import markov_clustering as mc
import numpy as np
import scipy.sparse as sp

from analysis_main.ents_base import EntityBase
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import DictionaryLearning


class EntityCompare(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres=10,
                 ignore_article_counts=True, precomp_ent_group=None):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts, precomp_ent_group)

    def compute_entity_vectors(self, word_count_mat, n_dim=300): 
        sim_mat = self._compute_sim_mat(word_count_mat)
        svd_comps = self._svd_decomp(sim_mat, n_dim)
        norm_vectors = StandardScaler().fit_transform(svd_comps).T
        self.sim_mat = sim_mat
        self.entity_vectors = norm_vectors
        
    def compute_sparse_entity_vectors(self, word_count_mat, n_dim=50):
        sim_mat = self._compute_sim_mat(word_count_mat)
        sparse_vecs = self._sparse_dict_decomp(sim_mat.toarray(), n_dim)
        return sparse_vecs
            
    def query_nearest_neighbors(self, query, voc2id):
        if not hasattr(self, 'nbrs_model'):
            print('No nearest neighbor model detected, running initial model.'
                  'This may take a minute.')
            self.nbrs_model = NN(n_neighbors=10, algorithm='brute', metric='cosine').fit(
                self.entity_vectors)
        vocab = list(voc2id.keys())
        if query in vocab:
            distances, indices = self.nbrs_model.kneighbors(
                self.entity_vectors[voc2id[query], :].reshape(-1, 1).T)
            for dist, indx in zip(distances[0], indices[0]):
                print('{} : {} \n'.format(vocab[indx], dist))
        else:
            print('query token does not exist in vocabulary')

    def _compute_sim_mat(self, word_count_mat, norm=False):
        word_cc = word_count_mat.T * word_count_mat
        word_cc.setdiag(0)
        # Compute PMI Matrix
        sim_mat = self._convert_to_ppmi_mat(word_cc, norm)
        return sim_mat
    
    @staticmethod
    def _convert_to_ppmi_mat(word_cc, norm):
         # total word counts
        Z = word_cc.sum()

        # counts per article.
        Zr = np.array(word_cc.sum(axis=1), dtype=np.float64).flatten()

        # Get indices of non zero elements
        ii, jj = word_cc.nonzero()  # row, column indices
        Cij = np.array(word_cc[ii,jj]).flatten()

        # calc positive PMI
        pmi = np.log( (Cij * Z) / (Zr[ii] * Zr[jj]))
        if norm:
            pmi = pmi / -np.log(Cij * Z)
        ppmi = np.maximum(0, pmi)  # take positive only
        # Create sparse matrix
        ppmi_mat = sp.csc_matrix((ppmi, (ii,jj)), shape=word_cc.shape,
                                      dtype=np.float64)
        ppmi_mat.eliminate_zeros()  

        return ppmi_mat    
    
    def _svd_decomp(self, sim_mat, n_comp):
        trunc_svd = TruncatedSVD(n_components=n_comp)
        trunc_svd.fit(sim_mat)
        normed_comps = StandardScaler().fit_transform(trunc_svd.components_.T).T
        return normed_comps