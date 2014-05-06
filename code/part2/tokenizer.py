from porter_stemmer import PorterStemmer
import re
import string

class Tokenizer(object):
    def __init__(self, stopwords_file):
        self.stemmer = PorterStemmer()
        self.stop_words = set(line.strip() for line in open(stopwords_file))

    def __call__(self, text):
        tokens = []
        text = re.sub(r'[^\w\s]', ' ', text) # replace punctuation with whitespace
        words = text.lower().split()
        for word in words:
            word = self.stemmer.stem(word, 0, len(word) - 1) # stem

            # throw out words in stop words and those starting with non alphabet
            if word and word not in self.stop_words and word[0].isalpha():
                tokens.append(word)

        return tokens

## Possible thing to add: replace 3 or more occurrences of the same character with two occurrences
# for sub_word in word.split():
#     # Replace three or more occurrences of the same character with two occurrences
#     unique_chars = ''.join(set(sub_word))
#     for c in unique_chars:
#         sub_word = re.sub(c + r'{3,}', c + c, sub_word)
