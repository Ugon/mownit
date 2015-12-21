from flask import Flask, render_template, request, send_from_directory
from query import NaiveQueryEngine, OptimizedQueryEngine
import argparse


app = Flask(__name__)
qe = None


@app.route('/')
def show_search_page():
    return render_template('show_search_page.html')


@app.route('/search')
def search():
    return render_template('result_page.html', results=qe.query(request.args['query'], int(request.args['number'])))


@app.route('/local/<name>')
def display_local_html(name):
    return send_from_directory('html-articles', name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-lsa', action='store_true')
    parser.add_argument('-idf', action='store_true')
    args = vars(parser.parse_args())
    lsa = args['lsa']
    idf = args['idf']

    if lsa and idf:
        qe = OptimizedQueryEngine('matrices/matrix-approx-idf.npy', 'matrices/matrix-u-transposed-idf.npy',
                                  'matrices/word-indices.txt', 'matrices/inverse-article-indices.txt',
                                  'various/urls.csv')
    elif lsa and not idf:
        qe = OptimizedQueryEngine('matrices/matrix-approx-non-idf.npy', 'matrices/matrix-u-transposed-non-idf.npy',
                                  'matrices/word-indices.txt', 'matrices/inverse-article-indices.txt',
                                  'various/urls.csv')
    elif not lsa and idf:
        qe = NaiveQueryEngine('matrices/matrix-full-idf.npz', 'matrices/word-indices.txt',
                              'matrices/inverse-article-indices.txt', 'various/urls.csv')
    elif not lsa and not idf:
        qe = NaiveQueryEngine('matrices/matrix-full-non-idf.npz', 'matrices/word-indices.txt',
                              'matrices/inverse-article-indices.txt', 'various/urls.csv')

    app.run()
