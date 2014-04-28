import sys
import argparse
import json
import csv
import re

from collections import defaultdict
from tokenizer import Tokenizer

def main():
    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reviews', required=True, help='Review data file')
    parser.add_argument('-o', '--out', required=True, help='Inverted index output file')
    parser.add_argument('-s', '--stop', required=True, help='Stopword list')
    opts = parser.parse_args()

    ## Output file
    csv_writer = csv.writer(open(opts.out, 'w'), delimiter="\t")
    csv_writer.writerow(['token', 'business_id', 'review_id', 'position', '...'])

    ## Tokenizer
    tk = Tokenizer(opts.stop)
    token_map = defaultdict(list)

    ## Tokenize review texts
    # for each word in the vocabulary (in this case all words found in all reviews):
    # business id, review id, and position of each term occurrence
    # instead of using the review id, uses the line on which the review occurs as a unique identifier
    reviews = open(opts.reviews)
    for review_num, line in enumerate(reviews):
        review = json.loads(line)
        business_id = review['business_id'].encode('utf-8')
        tokens = tk.tokenize(review['text'])
        for position, word in enumerate(tokens):
            token_map[word].append((business_id, review_num, position))

    ## Print sorted inverted index
    for token in sorted(token_map):
        row = [token]
        row.extend(token_map[token])
        csv_writer.writerow(row)

if __name__ == '__main__':
	main()
