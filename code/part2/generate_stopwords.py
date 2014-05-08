from __future__ import division
from tokenizer import Tokenizer
import sys
from collections import defaultdict
import json
import pdb

def generate_stopwords(review_file, business_file, outfile, stopwords_file):
	tokenizer = Tokenizer(stopwords_file)
	occurances = defaultdict(int)
	with open(review_file) as review_file:
		for i, review in enumerate(review_file):
			data = json.loads(review)
			line = tokenizer(data['text'])
			for token in line:
				occurances[token] += 1
			if i %1000 == 0:
				print i
	sorted_tokens = sorted(occurances.items(), key = lambda item: item[1])
	def shouldIgnore(item):
		word, count = item
		if count <= 3:
			return True
		if count > 50000:
			return True
		return False

	output = set(word for word, count in filter(shouldIgnore, sorted_tokens))
	
	with open(outfile, "w") as towrite:
		towrite.write("\n".join(output))


if __name__ == "__main__":
	review_file, business_file, outfile,stopwords_file = sys.argv[1:]
	generate_stopwords(review_file, business_file, outfile,stopwords_file)

