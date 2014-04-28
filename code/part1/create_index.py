import sys
import argparse
import json
import csv
import re

from collections import defaultdict
from porter_stemmer import PorterStemmer

def main():
    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reviews', required=True, help='Review data file')
    parser.add_argument('-o', '--out', required=True, help='Inverted index output file')
    parser.add_argument('-s', '--stop', required=True, help='Stopword list')
    opts = parser.parse_args()

    '''
    Review json format
    {
        'type': 'review',
        'business_id': (encrypted business id),
        'user_id': (encrypted user id),
        'stars': (star rating, rounded to half-stars),
        'text': (review text),
        'date': (date, formatted like '2012-03-14'),
        'votes': {(vote type): (count)},
    }
    '''

    ## Output file
    csv_writer = csv.writer(open(opts.out, 'w'), delimiter="\t")
    csv_writer.writerow(['token', 'business_id', 'review_id', 'position', '...'])

    ## Stemmer, stopwords, dict
    stopwords = set(line.strip() for line in open(opts.stop)) # Create stopword set
    stemmer = PorterStemmer() # init Stemmer
    token_map = defaultdict(list)

    ## Tokenize review texts
    # for each word in the vocabulary (in this case all words found in all reviews)
    # you should be able to retrieve a list of postings containing:
    # business id, review id, and position of each term occurrence
    # instead of using the review id, use the line on which the review occurs as a unique identifier

    reviews = open(opts.reviews)
    for review_num, line in enumerate(reviews):
        review = json.loads(line)
        business_id = review['business_id'].encode('utf-8')
        text = review['text'].lower() # lowercase

        for position, word in enumerate(text.split()):
            word = re.sub(r'[^\w\s]', '', word)
            word = stemmer.stem(word, 0, len(word) - 1) # apply stemming
            if word not in stopwords and word != '': # filter stopwords
                token_map[word].append((business_id, review_num, position))

    ## Print sorted inverted index
    for token in sorted(token_map):
        row = [token]
        row.extend(token_map[token])
        csv_writer.writerow(row)

if __name__ == '__main__':
	main()
