from scipy.sparse import csc_matrix, csr_matrix
from stemming.porter2 import stem
import numpy as np
import pickle
import re


def read_sparse_matrix(file_name):
    loader = np.load(file_name)
    return csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def read_matrix(file_name):
    return np.asmatrix(np.load(file_name))


def read_dict(file_name):
    with open(file_name, 'r') as f:
        d = pickle.load(f)
    return d


def build_query_normalized_sparse_vector(list_of_words, word_indices):
    stemmed = filter(lambda x: x in word_indices.keys(), map(stem, map(lambda x: re.sub('[^a-z]+', '', x.lower()), list_of_words)))
    vector = csr_matrix((len(word_indices), 1), dtype=np.float32)
    sum_of_squares = len(stemmed)
    n = 1 / np.sqrt(sum_of_squares) if sum_of_squares != 0 else 0
    for word in stemmed:
        vector[word_indices[word], 0] = n
    return vector


def build_query_normalized_low_rank_approximated_vector(list_of_words, word_indices, u_transposed):
    stemmed = filter(lambda x: x in word_indices.keys(), map(stem, list_of_words))
    vector = csr_matrix((len(word_indices), 1), dtype=np.float32)
    for word in stemmed:
        vector[word_indices[word]] = 1
    vector = u_transposed * vector
    sum_of_squares = sum(map(lambda x: x**2, vector))
    n = np.sqrt(sum_of_squares) if sum_of_squares != 0 else 1
    for a in vector.nonzero()[0]:
        vector[a] /= n
    return vector


def correlation(vector, column):
    return np.absolute((vector.T * column)[0, 0])


def correlations_vector(vector, matrix):
    return map(lambda x: correlation(vector, x), map(matrix.getcol, xrange(matrix.shape[1])))


def correlations_svd_vector(vector, matrix):
    return map(lambda x: correlation(vector, x), map(lambda y: matrix[:, y], xrange(matrix.shape[1])))


class NaiveQueryEngine:
    def __init__(self, matrix_file_name, word_indices_file_name, inverse_article_indices_file_name, url_file_name):
        self.matrix = read_sparse_matrix(matrix_file_name)
        self.word_indices = read_dict(word_indices_file_name)
        self.inverse_article_indices = read_dict(inverse_article_indices_file_name)
        self.urls = dict()
        with open(url_file_name, 'r') as f:
            for name, url in map(lambda x: x.split(';'), f.readlines()):
                self.urls[name] = url.rstrip()

    def query(self, query_string, num_of_results):
        list_of_words = query_string.split()
        query_vector = build_query_normalized_sparse_vector(list_of_words, self.word_indices)
        correlations_v = correlations_vector(query_vector, self.matrix)
        result_article_indices = np.argsort(correlations_v)[::-1]
        result_article_names = map(lambda x: self.inverse_article_indices[x].replace('.csv', ''),
                                   result_article_indices[:num_of_results])
        result = map(lambda x: {'name': x, 'file_name': x + '.html', 'url': self.urls[x]},
                     result_article_names)
        return result


class OptimizedQueryEngine:
    def __init__(self, matrix_approx_file_name, matrix_u_transposed_file_name,
                 word_indices_file_name, inverse_article_indices_file_name, url_file_name):
        self.matrix_approx = read_matrix(matrix_approx_file_name)
        self.u_transposed = read_matrix(matrix_u_transposed_file_name)
        self.word_indices = read_dict(word_indices_file_name)
        self.inverse_article_indices = read_dict(inverse_article_indices_file_name)
        self.urls = dict()
        with open(url_file_name, 'r') as f:
            for name, url in map(lambda x: x.split(';'), f.readlines()):
                self.urls[name] = url.rstrip()

    def query(self, query_string, num_of_results):
        list_of_words = query_string.split()
        query_vector = build_query_normalized_low_rank_approximated_vector(list_of_words, self.word_indices,
                                                                                  self.u_transposed)
        correlations_v = correlations_svd_vector(query_vector, self.matrix_approx)
        result_article_indices = np.argsort(correlations_v)[::-1]
        result_article_names = map(lambda x: self.inverse_article_indices[x].replace('.csv', ''),
                                   result_article_indices[:num_of_results])
        result = map(lambda x: {'name': x, 'file_name': x + '.html', 'url': self.urls[x]},
                     result_article_names)
        return result