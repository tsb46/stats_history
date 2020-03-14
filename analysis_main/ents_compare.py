import numpy as np
import scipy.sparse as sp

from analysis_main.ents_base import EntityBase
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.decomposition import TruncatedSVD


class EntityCompare(EntityBase):
    def __init__(self, ents_dir, journal_classifier, ent_count_thres,
                 ignore_article_counts=True, n_dim=300):
        super().__init__(ents_dir, journal_classifier, ent_count_thres,
                         ignore_article_counts)
        self.n_dim = n_dim

    def compute_entity_vectors(self, word_count_mat):
        sim_mat = self._compute_sim_mat(word_count_mat)
        svd_res = self._svd_decomp(sim_mat, self.n_dim)
        norm_vectors = Normalizer().fit_transform(svd_res.components_)
        self.entity_vectors = norm_vectors

    def query_nearest_neighbors(self, query, voc2id):
        if not hasattr(self, 'nbrs_model'):
            print('No nearest neighbor model detected, running initial model.'
                  'This may take a minute.')
            self.nbrs_model = NN(n_neighbors=10, algorithm='ball_tree').fit(
                self.entity_vectors.T)
        vocab = list(voc2id.keys())
        if query in vocab:
            distances, indices = self.nbrs_model.kneighbors(
                self.entity_vectors.T[voc2id[query], :].reshape(-1, 1).T)
            for dist, indx in zip(distances[0], indices[0]):
                print('{} : {} \n'.format(vocab[indx], dist))
        else:
            print('query token does not exist in vocabulary')

    def _compute_sim_mat(self, word_count_mat):
        # Compute word co-occurence matrix
        word_cc = word_count_mat.T * word_count_mat
        # Compute PMI Matrix
        sim_mat = self._convert_to_ppmi_mat(word_cc)
        return sim_mat

    @staticmethod
    def _convert_to_ppmi_mat(word_count_mat, smooth_alpha=0.75, positive_thres=True):
        num_cc = word_count_mat.sum()  # num of co-occurrences
        # set smoothing parameters
        nca_denom = np.sum(np.array(word_count_mat.sum(axis=0))
                           .flatten() ** smooth_alpha)
        sum_over_words = np.array(word_count_mat.sum(axis=0)).flatten()
        sum_over_words_alpha = sum_over_words ** smooth_alpha
        sum_over_contexts = np.array(word_count_mat.sum(axis=1)).flatten()
        # set up vars for sparse matrix
        row_indxs = []
        col_indxs = []
        pmi_dat_values = []
        coo_mat = word_count_mat.tocoo()
        for ii, jj, count in zip(coo_mat.row, coo_mat.col, coo_mat.data):
            # Get Terms for pair-wise PMI calc
            nwc = count
            Pwc = nwc / num_cc
            nw = sum_over_contexts[ii]
            Pw = nw / num_cc
            nc = sum_over_words[jj]
            Pc = nc / num_cc
            # Calculate PMI (type based on input parameters)
            if smooth_alpha > 0:
                nca = sum_over_words_alpha[jj]
                Pca = nca / nca_denom
                if positive_thres:
                    pmi = max(np.log2(Pwc / (Pw * Pca)), 0)
                else:
                    pmi = np.log2(Pwc / (Pw * Pca))
            else:
                if positive_thres:
                    pmi = max(np.log2(Pwc / (Pw * Pc)), 0)
                else:
                    pmi = np.log2(Pwc / (Pw * Pc))

            # Assign values for Sparse Matrix
            row_indxs.append(ii)
            col_indxs.append(jj)
            pmi_dat_values.append(pmi)

        # Create Sparse Positive Mutual Information Matrix
        ppmi_mat = sp.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
        return ppmi_mat

    def _svd_decomp(self, sim_mat, n_comp):
        trunc_svd = TruncatedSVD(n_components=n_comp)
        trunc_svd.fit(sim_mat)
        return trunc_svd