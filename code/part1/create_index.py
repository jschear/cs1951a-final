import sys
import argparse
import json

def main():
    ## args
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reviews', required=True, help='Review data file')
    parser.add_argument('-o', '--out', required=True, help='Inverted index output file')
    parser.add_argument('-s', '--stop', help='Stopword list')
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

    ## Tokenize review texts
    # for each word in the vocabulary (in this case all words found in all reviews)
    # you should be able to retrieve a list of postings containing:
    # business id, review id, and position of each term occurrence
    # instead of using the review id, use the line on which the review occurs as a unique identifier
    reviews = open(opts.reviews)
    for review_num, line in enumerate(reviews):
        review = json.loads(line)

        text = review['text']
        business_id = review['business_id']

        print text, business_id, review_num

        # lowercase

        # apply stemming

        # filter out stop words

        # print review


if __name__ == '__main__':
	main()
