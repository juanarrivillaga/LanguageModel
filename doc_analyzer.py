from nltk.stem.snowball import EnglishStemmer
import nltk
import json
import os
import os.path
import re
import operator
import math
import copy
import structures.document
from structures.ngram_trie import NgramTrie
from structures.node import Node
from structures.language_model import LanguageModel



class DocAnalyzer(object):
    """
    This class contains various methods for building and preprocessing
    document (python dictionary) objects. Assumes a directory of .json files. Each file
    contains all the review documents for a specific business on Yelp, and it
    is named by its unique ID on Yelp, e.g., FAhx3UZtXvqNiEAd-GNruQ.json;
    All the files are in json format. Each json file contains a json array of
    reviews ['Reviews'] and a json object about the information of the
    restaurant ['RestaurantInfo'].


    1 The json object for user review is defined as follows:

    {
        'Author':'author name (string)',
        'ReviewID':'unique review id (string)',
        'Overall':'numerical rating of the review (float)',
        'Content':'review text content (string)',
        'Date':'date when the review was published',
        'Author_Location':'author's registered location'
    }

    2 The json object for restaurant info is defined as follows:

    {
        'RestaurantID':'unique business id in Yelp (string)',
        'Name':'name of the business (string)',
        'Price':'Yelp defined price range (string)',
        'RestaurantURL':'actual URL to the business page on Yelp (string)',
        'Longitude':'longitude of the business (double)',
        'Latitude':'latitude of the business (double)',
        'Address':'address of the business (string)',
        'ImgURL':'URL to the business's image on Yelp (string)'
    }
    """

    def __init__(self, n, stopwords_file = 'data/english.stop.txt', stopwords = False):

        self.n = n # as in N-gram

        #self.docs = [] # here will go a list of reviews

        # tokenizer and stemmer
        self._tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        #self._stemmer = EnglishStemmer()

        # here will be a trie structure to hold n_gram counts
        self.stats = NgramTrie()

        # regexs used in normalization method
        self._punct_regex = re.compile(
                r"[`;!’[\](){}⟨⟩:,،‒–—―….?\“\”‘\"/⁄]+")
        self._num_regex = re.compile(r"[-.0-9]+")

        # here we will store document frequencies
        self._df = {}

        # finally, load stopwords.
        self._stopwords = set() # a set of stopwords to be loaded from a file
        if stopwords:
            self._load_stopwords(stopwords_file)
            self._stopwords.add('')

    def _load_stopwords(self, filename):
        """
        Loads list of stopwords from a file and adds them to the set
        self_stopwords.

        Args:
            filename(str) : the location of the file containing the stopwords
        """

        with open(filename) as f:
            for word in f:
                word = word.strip()
                word = self._normalize(word)
                word = self._stemmer.stem(word)
                self._stopwords.add(word)

        # now add the stopwords we discovered in our first pass (top 100 by DF)
        with open('data/restaurant.stop.txt') as f:
            for word in f:
                word = word.strip()
                self._stopwords.add(word)

    def _analyze_document(self, document, train = False):
        """
        This function wraps all the preprocessing and analysis done to a
        document and will be used in the load_directory method. Side effects of
        updating self.docs self.stats if train = True, otherwise, it only
        processes the document

        Args:
            document (dict): loaded from j_son
            train (boolean): if true, updates self.docs and self.stats
        Returns:
            A document with updated counts
        """
        # doc = structures.document.Document(document) +++++++++++++++++++
        # first normalize, then tokenize
        tokens = self._tokenizer.tokenize(self._normalize(document['Content']))

        for n in range(1, self.n + 1):
            for i in range(len(tokens) + 1 - n):
                ngram = self.stats.add_ngram(*tokens[i:i+n])
                ngram.value += 1


    def _from_json(self, json_file_name):
        """
        Here we do the json housekeeping...
        Args:
            json_file_name: the location of the file
        Returns:
            A decoded json object i.e. a python dictionary or None if any
            errors occurred when opening the file or parsing the .json
        """

        try:
            with open(json_file_name) as f:
                json_object = json.load(f)
        except UnicodeDecodeError as e:
            print(e.encoding, 'raised an error.')
            print(e.reason)
            json_object = None
        except IOError as e:
            print("Failed to open", json_file_name)
            print(e.msg)
            json_object = None
        except json.decoder.JSONDecodeError  as e:
            print(e.msg)
            print("Deserializing:", e.doc,
                    "is not a valid json document. Parse error at",
                    e.lineno,e.colno)
            json_object = None

        return json_object

    def load_directory(self, directory, suffix = '.json', train = True):
        """
        Loads and processes the .json files from a directory (and all children
        recursively). Should update self.docs, self.stats,

        Args:
            directory (str): location of directory
            suffix (str): will only open files with this suffix. Defaults to
                .json
        """

        directory = os.path.abspath(directory)

        for root, subdirs, files in os.walk(directory):
            print("--------- In:",root,"----------")
            # filter out files that don't end with suffix
            for doc in filter(lambda s:s.endswith(suffix), files):
                print("Loading:", doc)
                jobject = self._from_json(os.path.join(root,doc))
                # check for if json loaded properly
                if jobject:
                    # iterate through reviews to feed to analze_document
                    for review in jobject['Reviews']:
                        self._analyze_document(review, train = True)
        # calculate document frequency in the training documents
        # self._calc_doc_frequency()

    def create_language_model(self, n):
        lm = LanguageModel(n, self.stats)
        self._mle_from_ref(0, self.stats.root, n)
        return lm


    def _mle_from_ref(self, depth, ref, n, prefix_count = 1):
        # maximum likelihood estimate
        ref.mle = ref.value / prefix_count


        # only continue updating as long as depth is less than or equal to n
        if (depth + 1) <= n:
            # total_value is precisely the observed frequency
            # of n_grams that start with the prefix represented by this node
            total_value = sum(ref.children[k].value for k in ref.children)

            # now recursively update the child nodes
            for k in ref.children:
                # create node object to add to dictionary
                self._mle_from_ref(depth + 1,
                        ref.children[k], n, prefix_count = total_value)

    def _normalize(self, token):
        """
        Performs normalization: remove punctuation, make all letters lowercase,
        replace integer or decimal number with special token 'NUM'.

        Args:
            token(str): a string to be normalized
        Returns:
            a normalized string
        """
        normalized = token
        # remove punctuation
        normalized = self._punct_regex.sub('', normalized)
        # make all of the letters lowercase
        normalized = normalized.lower()
        # replace integer or decimal number with
        normalized = self._num_regex.sub('NUM', normalized)

        return normalized

    def process_doc(self, doc):
        return self._tokenizer.tokenize(self._normalize(doc))


if __name__ == '__main__':
    analyzer = DocAnalyzer(2)
    analyzer.load_directory('data/yelp/train')
    root, subdirs, files = next(os.walk('/home/juan/workspace/Spring2016/TextMining/Assignment1/project/data/yelp/test'))
    test = []
    documents = []
    for file in files:
        print('Loading ', file)
        jobject = analyzer._from_json(os.path.join(root, file))
        for review in jobject['Reviews']:
            documents.append(review['Content'])
