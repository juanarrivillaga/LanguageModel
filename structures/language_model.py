from structures.ngram_trie import NgramTrie
import random
import math
class LanguageModel(object):

    def __init__(self, n, reference):
        self.n = n # order of n_gram
        self.lambda_ = 0.9
        self.delta = 0.1
        self.ref = reference # pointer to doc_analyzer.stats

    def linear_interpolation(self, *n_gram):
        if len(n_gram) <= 1:
            numerator = self.count(*n_gram) + 0.1
            denominator = self.ref.num_tokens[1] + 0.1*self.ref.num_types[1]
            return numerator / denominator
        else:
            return (self.lambda_*self.calc_mle(*n_gram) +
                    (1 - self.lambda_)*self.linear_interpolation(*n_gram[1:]))
    def absolute_discount(self, *n_gram):
        if len(n_gram) <= 1:
            numerator = self.count(*n_gram) + 0.1
            denominator = self.ref.num_tokens[1] + 0.1*self.ref.num_types[1]
            return numerator / denominator
        else:
            # delta * S where S = seen word types after n_gram[:-1]
            try:
                deltaS = 0.1*len(self.ref.get_ngram(*n_gram[:-1]).children)
                lamb = deltaS/self.ref.get_value(*n_gram[:-1])
            except KeyError as e:
                lamb = 1
            try:
                value = (
                    max(self.count(*n_gram) - self.delta, 0) /
                    self.count(*n_gram[:-1]) +
                    lamb*self.absolute_discount(*n_gram[1:])
                  )
            except ZeroDivisionError as e:
                value = self.absolute_discount(*n_gram[1:])
            return value

    def calc_mle(self, *n_gram):
        observed = self.ref.get_mle(*n_gram)
        if observed:
            return observed
        order = len(n_gram)
        # if unseen, Types / (Types + Tokens)  from eq. 6.16 in Jurafsky Martin
        return 1 /(self.ref.num_types[order]
                + self.ref.num_tokens[order])

    def count(self, *n_gram):
        observed = self.ref.get_value(*n_gram)
        if observed:
            return observed
        return 0

    def sample(self, smoothing_func, *prior):
        smooth = smoothing_func
        prob = random.random()
        if prior:
            distribution = {word:smooth(*prior,word) for word in
                self.mle.get_ngram(*prior).children}
        else:
            distribution = {word:smooth(word) for word in
                self.mle.root.children}

        for word in distribution:
            prob -= distribution[word]
            if prob <= 0:
                return word

    def loglikelihood(self, word_list, smooth_func, ngram_order):
        if ngram_order == 1:
            return sum(math.log(smooth_func(w)) for w in word_list)
        p = math.log(smooth_func(word_list[0]))
        i = ngram_order - 1
        while i < len(word_list):
            p = p + math.log(smooth_func(*word_list[i-ngram_order + 1:i]))
            i += 1
        return p

    def perplexity(self, document_tokenized, smooth_func, order):
        LL = -1*self.loglikelihood(document_tokenized,smooth_func,order)/len(document_tokenized)
        return math.exp(LL)
