from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from collections import Counter
import numpy as np
import os
import pickle


def save_matrices(dir_name, matrix_full_non_idf, matrix_full_idf, matrix_approx_non_idf, matrix_approx_idf,
                  u_transposed_non_idf, u_transposed_idf, word_indices, inverse_word_indices,
                  article_indices, inverse_article_indices):
    print 'Saving matrices to file'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.savez(dir_name + '/matrix-full-idf', data=matrix_full_idf.data, indices=matrix_full_idf.indices,
             indptr=matrix_full_idf.indptr, shape=matrix_full_idf.shape)
    np.save(dir_name + '/matrix-approx-idf', matrix_approx_idf)
    np.save(dir_name + '/matrix-u-transposed-idf', u_transposed_idf)

    np.savez(dir_name + '/matrix-full-non-idf', data=matrix_full_non_idf.data, indices=matrix_full_non_idf.indices,
             indptr=matrix_full_non_idf.indptr, shape=matrix_full_non_idf.shape)
    np.save(dir_name + '/matrix-approx-non-idf', matrix_approx_non_idf)
    np.save(dir_name + '/matrix-u-transposed-non-idf', u_transposed_non_idf)

    with open(dir_name + '/word-indices.txt', 'w') as f:
        pickle.dump(word_indices, f)
    with open(dir_name + '/inverse-word-indices.txt', 'w') as f:
        pickle.dump(inverse_word_indices, f)
    with open(dir_name + '/article-indices.txt', 'w') as f:
        pickle.dump(article_indices, f)
    with open(dir_name + '/inverse-article-indices.txt', 'w') as f:
        pickle.dump(inverse_article_indices, f)


def build_sparse_term_document_matrix(bags_of_words_dir, dictionary_file_name, idf):
    word_indices = dict()
    inverse_word_indices = dict()
    article_indices = dict()
    inverse_article_indices = dict()
    num_of_articles = 0
    num_of_words = 0

    print '\tCalculating word indices'
    with open(dictionary_file_name, 'r') as dictionary_file:
        for word in map(lambda x: x.rstrip(), dictionary_file.readlines()):
            word_indices[word] = num_of_words
            inverse_word_indices[num_of_words] = word
            num_of_words += 1

    print '\tCalculating article indices'
    for article in os.listdir(bags_of_words_dir):
        article_indices[article] = num_of_articles
        inverse_article_indices[num_of_articles] = article
        num_of_articles += 1

    if idf:
        print '\tPreparing IDF'
        document_count = len(os.listdir(bags_of_words_dir))
        word_document_count = Counter()
        idfs = dict()

        for name in os.listdir(bags_of_words_dir):
            with open(bags_of_words_dir + '/' + name, 'r') as f:
                for word, count in map(lambda x: x.split(';'), f.readlines()):
                    word_document_count[word] += 1

        def idf_fun(w):
            return np.log(float(document_count) / float(word_document_count[w]))

        for word in word_document_count.elements():
            idfs[word] = idf_fun(word)

    if idf:
        print '\tConstructing sparse matrix'
    else:
        print '\tConstructing sparse matrix with IDF applied'
    row = []
    col = []
    data = []

    for name in os.listdir(bags_of_words_dir):
        with open(bags_of_words_dir + '/' + name, 'r') as f:
            for word, count in map(lambda x: x.split(';'), f.readlines()):
                row.append(word_indices[word])
                col.append(article_indices[name])
                if idf:
                    val = float(count) * idfs[word]
                else:
                    val = float(count)
                data.append(val)

    matrix_coo = coo_matrix((data, (row, col)), shape=(num_of_words, num_of_articles), dtype=np.float32)

    return matrix_coo, word_indices, inverse_word_indices, article_indices, inverse_article_indices


def build_approx_components(matrix, rank):
    u, s, vt = svds(matrix, rank)
    return np.asmatrix(u.T), np.diag(s) * np.asmatrix(vt)


def normalize_sparse_matrix(matrix):
    print 'Normalizing column vectors in non-SVD sparse matrix'
    matrix = matrix.tocsc()
    sq = matrix.multiply(matrix)
    sums_of_squares = sq.sum(axis=0)

    data = []
    row = []
    col = []
    for (r, c) in zip(matrix.nonzero()[0], matrix.nonzero()[1]):
        row.append(r)
        col.append(c)
        data.append(1/np.sqrt(sums_of_squares[0, c]) if sums_of_squares[0, c] != 0 else 0)
    m2 = coo_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1)).tocsc()
    return matrix.multiply(m2)


def normalize_dense_matrix(matrix):
    print 'Normalizing column vectors in SVD dense matrix'
    sq = np.multiply(matrix, matrix)
    sums_of_squares = sq.sum(axis=0)
    for c in xrange(matrix.shape[0]):
        matrix[:, c] /= np.sqrt(sums_of_squares[0, c])

    return matrix


if __name__ == "__main__":
    print "Calculating term-document matrix with idf"
    mi, b, c, d, e = build_sparse_term_document_matrix('bags-of-words', 'various/dictionary.txt', idf=True)
    print "Calculating SVD for term-document matrix without idf"
    uti, approx_matrixi = build_approx_components(mi, int(3000))
    mi = normalize_sparse_matrix(mi)
    uti = normalize_dense_matrix(uti)

    print "Calculating term-document matrix without idf"
    mn, b, c, d, e = build_sparse_term_document_matrix('bags-of-words', 'various/dictionary.txt', idf=False)
    print "Calculating SVD for term-document matrix with idf"
    utn, approx_matrixn = build_approx_components(mn, int(3000))
    mn = normalize_sparse_matrix(mn)
    utn = normalize_dense_matrix(utn)

    save_matrices('matrices', mn, mi, approx_matrixn, approx_matrixi, utn, uti, b, c, d, e)
