from bs4 import BeautifulSoup
from stemming.porter2 import stem
from collections import Counter
import re
import os


def process_file(html_dir, name, stop_words):
    print "Processing: " + name
    f = open(html_dir + '/' + name, 'r')
    words = Counter()
    soup = BeautifulSoup(f.read())
    count = 0

    for word in map(lambda x: stem(x), filter(lambda x: len(x) > 1, map(lambda x: re.sub('[^a-z]+', '', x.lower()),
                                                                        soup.getText().split()))):
        if word not in stop_words:
            words[word] += 1
            count += 1

    weight = max(int(0.1 * count), 1)
    for word in map(lambda x: stem(x), filter(lambda x: len(x) > 1, map(lambda x: re.sub('[^a-z]+', '', x.lower()),
                                                                        name.split()))):
        if word not in stop_words:
            words[word] += weight

    f.close()
    return words


def process_files(html_dir, bags_of_words_dir, stop_words):
    if not os.path.exists(bags_of_words_dir):
        os.makedirs(bags_of_words_dir)

    all_words = set()
    for name in os.listdir(html_dir):
        i_file = open(bags_of_words_dir + '/' + name.replace('.html', '.csv'), 'w+')

        words = process_file(html_dir, name, stop_words)
        all_words |= set(words.elements())

        for (word, count) in sorted(words.items()):
            i_file.write(word + ';' + str(count) + '\n')

        i_file.close()

    all_words_file = open('various/dictionary.txt', 'w+')
    for word in sorted(all_words):
        all_words_file.write(word + '\n')


if __name__ == "__main__":
    sw = open('resources/stop-words.txt', 'r')
    stop_words_set = set(map(lambda word: word.rstrip(), sw.readlines()))
    process_files('html-articles', 'bags-of-words', stop_words_set)
