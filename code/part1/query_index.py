import sys
import csv
import argparse
from tokenizer import Tokenizer
from collections import defaultdict
import pdb


class UninformativeQueryException(Exception):
    pass

class SearchEngine:
    def __init__(self, index_file, tokenizer, num_results = 10):
        self.tokenizer = tokenizer
        self.data = defaultdict(list)
        self.num_results = num_results
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
                self.data[token] = set(eval(tup) for tup in row[1:])
        print "processed"


    def process_query(self,query):
        raw_results = self.find_results(query)
        ranked_results = self.rank_results(raw_results)
        return self.strrep_results(ranked_results)


    def strrep_results(self,ranked_results):
        return "\n".join(str(item) for item in ranked_results[0:self.num_results])

    def rank_results(self,results):
        return results

    def find_results(self, query):
        if query[0] == "\"" == query[-1] == "\"":
            tokenized = tokenizer.tokenize(query[1:-1])
            if len(tokenized) == 0:
                raise UninformativeQueryException
            if len(tokenized) == 1:
                return self.one_word_query(tokenized[0])
            else:
                return self.phrase_query(tokenized)
        tokenized = self.tokenizer.tokenize(query)
        if len(tokenized) == 0:
            raise UninformativeQueryException
        if len(tokenized) == 1:
            return self.one_word_query(tokenized[0])
        else:
            return self.free_text_query(tokenized)
        # return {"OWQ" : one_word_query, "FTQ" : free_text_query, "PQ" : phrase_query}[self.query_type(query)](query)


    def one_word_query(self,tokenized_word):
        print "Running a one word query on" + str(tokenized_word)
        resultlist = self.data[tokenized_word]
        return list(resultlist)

    def free_text_query(self,tokenized_line):
        # first = tokenized_line[0]
        # for word in tokenized_line:

        pass

    def phrase_query(self,tokenized_phrase):
        print "phrase_query"
        pass




#python query_index.py -r ../../tmp/reviews_small.json -i ../../tmp/inverted_index_small.csv -s ./stopwords.txt
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', required=True, help='Search Index')
    parser.add_argument('-s', '--stop', required=True, help='Stopword list')
    parser.add_argument('-r', '--reviews', required = True, help = 'Review File')
    opts = parser.parse_args()
    # pdb.set_trace()
    tokenizer = Tokenizer(opts.stop)
    search_engine = SearchEngine(opts.index, tokenizer)
    while True:
        query = raw_input("Please enter search query:")
        print search_engine.process_query(query)

if __name__ == '__main__':
    main()

