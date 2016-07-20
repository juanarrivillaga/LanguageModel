import json

class Document(object):
    """
    This class is here simply for wrapping documents. The models are
    dictionaries from tokens to values (e.g. tf-idf). One method is
    similarity, which finds cosine similarity between this document and
    another that contains a vector model.
    """

    def __init__(self, j_review):
        self.author = j_review.get('Author')
        self.review_id = j_review.get('ReviewID')
        self.rating = j_review.get('Rating')
        self.content = j_review.get('Content')
        self.date = j_review.get('Date')
        self.location = j_review.get('Author_Location')
        self.vector_model = {}
        self.count = {}
        self._norm = 0

    def similarity(self, other_doc):
        """
        Returns:
            Cosine similarity between this document and another. Both
        must contain vector space models. ASSUMES BOTH MODELS COME
        FROM THE SAME TRAINING SET!!!
        """
        # get intersection of keys
        common_keys = set(self.vector_model) & set(other_doc.vector_model)

        # sum of product of common key values (tf-idf). This is dot product
        result = sum(self.vector_model[k]*other_doc.vector_model[k]
                for k in common_keys)

        # finally, divide by the product of the norms
        try:
            result = result / (self._norm * other_doc._norm)
        # some docs will have norms of zero .... just assign 0
        except ZeroDivisionError as e:
            result = 0
        return result
