import sys
import csv
import argparse
import tokenizer
from collections import defaultdict
import pdb

class SearchEngine:
    def __init__(self, index_file, tokenizer):
        self.tokenizer = tokenizer
        self.data = {}
        self.read_data(index_file)




    def read_data(self, index_file):
        with open(index_file) as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            header = reader.next()
            line = 0
            for row in reader:
                line += 1
                if line %1000 == 0:
                    print line
                token = row[0]
                self.data[token] = [eval(tup) for tup in row[1:]]
        print "processed"
        pdb.set_trace()



    def process_query(query):
        pass

    def one_word_query(self,word):

    def free_text_query(self,words):

    def phrase_query(self,pharse):





#python query_index.py -i ../../../data/output.csv -r ../../../data/extracted/yelp_academic_dataset_review.json -s ./stopwords.txt 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', required=True, help='Search Index')
    parser.add_argument('-s', '--stop', required=True, help='Stopword list')
    parser.add_argument('-r', '--reviews', required = True, help = 'Review File')
    opts = parser.parse_args()
    # pdb.set_trace()
    #tokenizer = tokenizer.Tokenizer(opts.stop)
    tokenizer = None
    search_engine = SearchEngine(opts.index, tokenizer)




    for line in sys.stdin:
        print process_query(line)

if __name__ == '__main__':
    main()

