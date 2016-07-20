from structures.node import Node
class NgramTrie(object):
    def __init__(self):
        self.root = Node()
        self.num_types = {}
        self.num_tokens = {}

    def add_ngram(self, *args):
        # start at root
        curr_node = self.root
        # walk down the trie adding nodes as necessary
        for word in args:
            new = curr_node.insert(word)
            curr_node = new if new else curr_node.get_child(word)

        if new:
            self.num_types[len(args)] = self.num_types.get(len(args), 0) + 1
        self.num_tokens[len(args)] = self.num_tokens.get(len(args), 0) + 1
        curr_node.is_ngram = True
        return curr_node

    def is_ngram(self, *args):
        curr_node = root
        for word in args:
            seen = curr_node.get_child(word)
            if not seen:
                return False
        return seen.is_ngram

    def get_value(self, *args):
        curr_node = self.root
        for word in args:
            nextnode = curr_node.get_child(word)
            if not nextnode:
                return None
            else:
                curr_node = nextnode
        return curr_node.value

    def get_mle(self, *args):
        curr_node = self.root
        for word in args:
            nextnode = curr_node.get_child(word)
            if not nextnode:
                return None
            else:
                curr_node = nextnode
        return curr_node.mle


    def get_ngram(self, *args):
        curr_node = self.root
        for word in args:
            nextnode = curr_node.get_child(word)
            if not nextnode:
                raise KeyError(word)
            curr_node = nextnode
        return nextnode
