from __future__ import division
import sys
import csv
import json
import argparse
from tokenizer import Tokenizer
from collections import defaultdict
import operator
import pdb
import math

class UninformativeQueryException(Exception):
    pass

class SearchEngineData(): #lazy access dictionary on disk, takes in an index, and a file descriptor (that stays open, yikes)
    def __init__(self, index, path, to_cache = {}): 
        self.data = defaultdict(lambda: defaultdict(list))
        self.index = index
        print to_cache, len(to_cache)
        # pdb.set_trace()
        self.f = open(path,"r")
        for i,item in enumerate(to_cache):
            print i
            self.__getitem__(item)

    def __getitem__(self, key): #returns 
        # pdb.set_trace()
       
        if key in self.data:
            return self.data[key]
        # print "getting" + key
        if key not in self.index:
            return defaultdict(list)
        pos = self.index[key]
        self.f.seek(pos)
        line = self.f.readline()
        # print "processing"
        return self.process_line(line)

    def process_line(self, line):
        row = line.split("\t")
        token = row[0]
        for item in row[1:]:
            businessID, review_num, position = eval(item) #which is review_num, and which is position again?
            self.data[token][businessID].append((review_num, position))
        # pdb.set_trace()
        # print "done processing"
        return self.data[token]

                    # self.data[token].append((businessID,review_num,position))  #4165709824B total usage



    # def __setitem__(self, key):

class SearchEngine:
    def __init__(self, index_file, tokenizer, business_file, num_results = 10, in_memory = False):
        self.tokenizer = tokenizer
        self.fields_to_display = ['name','business_id','full_address','stars','review_count','categories']
        self.CACHE_THRESHOLD = 100000000
        # self.data = defaultdict(lambda : defaultdict(list))
        # self.data = defaultdict(list)
        self.data = None
        self.num_results = num_results
        self.num_businesses = 0
        self.business_data = {}
        if not in_memory:
            self.read_data_index(index_file)
        else:
            self.read_data(index_file)

        self.read_business_file(business_file)

    def read_business_file(self, business_file):
         with open(business_file) as businesses:
            for line in businesses:
                doc = json.loads(line.encode('utf8',"replace"))
                business_id = doc['business_id']
                relevant_data = {}

                for item in self.fields_to_display:
                    relevant_data[item] = doc[item]

                self.business_data[business_id] = relevant_data
                # pdb.set_trace()
            self.num_businesses = len(self.business_data)

    def read_data(self, index_file):
        self.data = defaultdict(lambda: defaultdict(list))
        with open(index_file) as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            header = reader.next()
            line = 0
            for row in reader:
                line += 1
                if line %1000 == 0:
                    print line
                token = row[0]
                for item in row[1:]:
                    businessID, review_num, position = eval(item) #which is review_num, and which is position again?
                    businessID = intern(businessID)
                    self.data[token][businessID].append((review_num, position))


    def read_data_index(self, index_file):
        with open(index_file) as f:
            index = {}
            to_cache = set()
            f.readline()
            currindex = f.tell()
            for line in iter(f.readline,''):
                token = line.split("\t")[0]
                if len(line) > self.CACHE_THRESHOLD:
                    to_cache.add(token)
                index[token] = currindex
                currindex = f.tell()


            self.data = SearchEngineData(index, index_file,to_cache = to_cache)




        
                    # self.data[token].append((businessID,review_num,position))  #4165709824B total usage
            
            # for line in reviews:
            #     self.ratings[line] = json.loads(line)['stars']
            # pdb.set_trace()
            # pdb.set_trace()
       
            # pdb.set_trace()


            
                # self.data[token] = [eval(item) for item in row]



        print "processed"

    # def make_doc_dict(self,tuples): #takes the tuples of the form (businessID, review, frequency), outputs them to a dictionary like this: business : reviewID : list of positions 
    #     out = defaultdict(list)
    #     for businessID, review_num, position in tuples:
    #         out[businessID].append((review_num,position))
    #     return out


    def process_query(self,query,html = False):
        terms, businesses = self.find_results(query)
        ranked_results = self.rank_results(terms, businesses)
        filtered_results = self.filter_results(ranked_results)
        if not html:
            return self.strrep_results(filtered_results)
        else:
            return self.htmlrep_results(filtered_results)

    def htmlrep_business(self, business_id):
        strrep = self.strrep_business(business_id) + " <a href =\"WEBSITE\">link</a>"
        strrep = strrep.replace("\n","<br>")
        return strrep

    def strrep_business(self, business_id):
        def strrep_item(field_name, value):
            # field_name, value = item
            if field_name == "categories":
                return "<em> " + str(field_name) + "</em> : " + ", ".join(map(str, value))
            else:
                return "<em> " + str(field_name) + "</em> : " + str(value)
        data = self.business_data[business_id]
        # pdb.set_trace()
        fields = map(data.get, self.fields_to_display)
        # print zip(self.fields_to_display,fields)
        out = "\n".join(map(lambda item: strrep_item(*item), zip(self.fields_to_display, fields)))
        # out = "Name : " + name + "\nRating: " + str(stars) + "\nReview Count: " + str(review_count) + "\nBusinessID: " + business_id 
        out += "\n"
        # try:
        #     out = """
        #         Name: {name} 
        #         Rating: {stars}
        #         Review Count: {review_count}
        #         Business ID: {business_id}
        #         """.format(**data)
        # except Exception:
        #     pdb.set_trace()
        return out


    def strrep_results(self, business_ids):
        # pdb.set_trace()
        return "\n".join(map(self.strrep_business,business_ids))

    def htmlrep_results(self, business_ids):
        out = "<ol>"
        for business_id in business_ids:
            out += "<li>" + self.htmlrep_business(business_id) + "</li>"
        out += "</ol>"
        return out


    def tf_idf(self, business_id, token):
        # pdb.set_trace()
        tf = len(self.data[token][business_id])
        idf = math.log(self.num_businesses/len(self.data[token]))
        return tf*idf

    def get_rating(self, business_id):
        return self.business_data[business_id]['stars']

    def get_business_data(self, business_id):
        return self.business_data[business_id]

    def get_review_count(self, business_id):
        return self.business_data[business_id]['review_count']

    def ranking_function(self, business_id, terms):
        rating = self.get_rating(business_id)
        num_reviews = self.get_review_count(business_id)
        return rating*math.log(num_reviews)*sum(map(lambda term: self.tf_idf(business_id,term),terms))
        # return sum(map(lambda term: self.tf_idf(business_id,term),terms))

    def rank_results(self, business_ids, terms):
        return sorted(business_ids, key = lambda b: self.ranking_function(b,terms),reverse = True)

    def filter_results(self,business_ids):
        return business_ids[:self.num_results]

    def find_results(self, query):
        if query[0] == "\"" == query[-1] == "\"":
            tokenized = self.tokenizer.tokenize(query[1:-1])
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
        return [business_id for business_id in self.data[tokenized_word]], [tokenized_word] 

    def free_text_query(self,tokenized_line):
        return list(reduce(operator.and_ ,map(lambda token: self.data[token].viewkeys(), tokenized_line))), tokenized_line

    def phrase_query(self, tokenized_line):
        business_ids, _ = self.free_text_query(tokenized_line)
        # print len(business_ids)
        out = filter(lambda business_id: self.contains_phrase(business_id, tokenized_line), business_ids)
        # pdb.set_trace()
        # print len(out)

        return out, tokenized_line

    # def contains_phrase(self, business_id, tokenized_line):
    #     # pdb.set_trace()
    #     # for offset, word in enumerate(tokenized_line):
    #     #     pos_set = _offset_position_set(word,business_id, offset) #set of 
    #     pos_sets = [self._offset_position_set(word, business_id, offset) for offset, word in enumerate(tokenized_line)]
    #     return filter(lambda set_: len(set_) > 0, pos_sets)

    def contains_phrase(self, business_id, tokenized_line):
        pos_sets = [self._offset_position_dict(word, business_id, offset) for offset, word in enumerate(tokenized_line)] #list of dict(review -> (pos...)) lists
        # first_set, rest = pos_sets[0], pos_sets[1:]
        # for review in first_set: #for every review in the first set
        #     for set_ in rest: #for every other set
        #         if review not in set_: break #if that first review isn't in every other word-set, then break
        common_reviews = self.dict_intersect(pos_sets) #the reviews that are common for every word, filtering out the ones we don't need
        # print common_reviews
        # print tokenized_line
        # print len(common_reviews)
        # if business_id == "8KXIh45r0PO4C_eJffmNNA":
            # pdb.set_trace()
        for review in common_reviews: #for every common review
            review_word_sets = [ pos_set[review] for pos_set in pos_sets ]
            # for pos_set in pos_sets: #for every word:
            #     word_set = pos_set[review]
            #     review_word_sets.append(word_set)
            if len(self.set_intersect(review_word_sets)) > 0:
                return True
        return False

        # for pos_set in pos_sets: #for every word in the phrase
        #     for review in common_reviews
            



    def set_intersect(self, sets):
        return reduce(operator.and_, sets)

    def dict_intersect(self, dicts): #gets the list of keys for a given set
        # pdb.set_trace()
        return reduce(lambda d1, d2: set(d1) and set(d2), dicts)
    # def common_element(self, sets):
        # if len(sets) == 1:
        #     return True
        # else:
        #     starter, rest = sets[0], sets[1:]
        #     for item in starter:
        #         for set_ in rest:
        #             if item not in set_:
        #                 break



    def _offset_position_dict(self, word, business_id, offset): #turns a list of (review_num, position) [corresponding to a business_id and word], to a map from review_num to a set of positions - offset (to check consecutiveness)
        index_list = self.data[word][business_id]
        out = defaultdict(set)
        for review_num, position in index_list:
            out[review_num].add(position-offset)
        # pdb.set_trace()

        return out





#python query_index.py -i ../../tmp/inverted_index.csv -s ./stopwords.txt -b ../../tmp/businesses.json
#python query_index.py -i ../../tmp/inverted_index_small.csv -s ./stopwords.txt -b ../../tmp/businesses.json
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', required=True, help='Search Index')
    parser.add_argument('-s', '--stop', required=True, help='Stopword list')
    parser.add_argument('-b', '--businesses', required = True, help = 'Business listing file')
    parser.add_argument('-m', '--in_memory', action = "store_true", help = 'Flag to store all the data in memory. Increases program speed memory consumption, decreases the startup speed.')
    parser.add_argument('-html', '--htmlrep', action = "store_true", help = 'Display results as an html list')
    opts = parser.parse_args()
    # pdb.set_trace()
    tokenizer = Tokenizer(opts.stop)
    search_engine = SearchEngine(opts.index, tokenizer, opts.businesses, num_results = 10, in_memory = bool(opts.in_memory))
    while True:
        try:
            query = raw_input("Please enter search query:")
            print search_engine.process_query(query, opts.htmlrep)
        except EOFError:
            return

if __name__ == '__main__':
    main()

