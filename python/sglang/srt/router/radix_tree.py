from typing import List

class Node:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.ids: List[int] = []
        self.count = 0

class RadixTree:
    def __init__(self):
        # dummy root
        self.root = Node()

    def insert(self, ids):
        curr = self.root
        unmatched_ids = ids

        while len(unmatched_ids) > 0:
            has_match = False

            for child in curr.children:
                prefix_len = self._ids_prefix_match(unmatched_ids, child.ids)

                if prefix_len == 0: continue
                    
                if prefix_len <= len(unmatched_ids):
                    unmatched_ids = unmatched_ids[prefix_len:]
                    child.count += 1
                    curr = child
                    has_match = True
                    break
                else: # if prefix_len > len(unmatched_ids)
                    # split the child
                    new_node = Node()
                    new_node.ids = child.ids[prefix_len:]
                    new_node.count = child.count + 1
                    new_node.children = child.children
                    child.children = [new_node]
                    unmatched_ids = []
                    curr = new_node
                    has_match = True
                    break

            if has_match is False:
                new_node = Node()
                new_node.ids = unmatched_ids
                new_node.count = 1
                curr.children.append(new_node)
                curr = new_node
                unmatched_ids = []
    
    def _ids_prefix_match(self, a, b):
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                return i
        return min(len(a), len(b))

    def pretty_print(self):
        ...


    def match(self, ids):
        ...

    def delete(self, ids):
        ...


