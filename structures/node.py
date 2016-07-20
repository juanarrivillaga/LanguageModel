class Node(object):
    __slots__ = ('value', 'mle', 'children', 'is_ngram')
    def __init__(self, word = None):
        self.value = 0
        self.mle = 0
        self.children = {}
        self.is_ngram = False
    def get_child(self, token):
        return self.children.get(token)
    def insert(self, word):
        if word in self.children:
            return None
        else:
            new_node = Node()
            self.children[word] = new_node
            return new_node
