from typing import List

class Node:
    def __init__(self):
        self.children = []
        self.is_end = False
        self.ids: List[int] = []
        self.count = 0

class RadixTree:
    def __init__(self):
        # dummy root
        self.root = Node()

    def insert(self, ids):
        curr = self.root
        curr.count += 1
        unmatched_ids = ids

        while len(unmatched_ids) > 0:
            has_match = False

            for child in curr.children:
                prefix_len = self._ids_prefix_match(unmatched_ids, child.ids)

                if prefix_len == 0: continue
                    
                if len(child.ids) == prefix_len:
                    unmatched_ids = unmatched_ids[prefix_len:]
                    child.count += 1
                    curr = child
                    has_match = True
                    break
                else: # if len(child.ids) > prefix_len
                    # split the child
                    new_node = Node()
                    new_node.ids = child.ids[prefix_len:]
                    child.ids = child.ids[:prefix_len]

                    new_node.children = child.children
                    child.children = [new_node]

                    new_node.count = child.count

                    unmatched_ids = unmatched_ids[prefix_len:]
                    child.count += 1
                    curr = child # continue from the child
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
        """
        Prints the RadixTree in a hierarchical format.
        Each level is indented by 2 spaces.
        Shows the node's IDs and count.
        """
        self._pretty_print_helper(None, 0)

    def _pretty_print_helper(self, node, level):
        if node is None:
            node = self.root
            print("Root")
            
        indent = "  " * level
        
        for child in node.children:
            print(f"{indent}├─ ids:{child.ids} (count:{child.count})")
            self._pretty_print_helper(child, level + 1)

    def prefix_match(self, ids):
        """
        Return the matched prefix
        """
        curr = self.root
        match_end_idx = 0
        ids_len = len(ids)
        
        while match_end_idx < ids_len:
            has_full_match = False
            unmatched_ids = ids[match_end_idx:]

            for child in curr.children:
                prefix_len = self._ids_prefix_match(unmatched_ids, child.ids)
                if prefix_len == 0: continue
                if len(child.ids) == prefix_len:
                    match_end_idx += prefix_len
                    curr = child
                    has_full_match = True
                    break
                else: # len(child.ids) > prefix_len
                    match_end_idx += prefix_len
                    break
                    
            if not has_full_match:
                break

        return ids[:match_end_idx]

    def delete(self, ids):
        # guarantee ids must be in the tree
        if self.prefix_match(ids) != ids:
            raise ValueError("IDs not found in the tree")
        
        curr = self.root
        curr.count -= 1
        match_end_idx = 0
        ids_len = len(ids)

        while match_end_idx < ids_len:
            unmatched_ids = ids[match_end_idx:]
            has_full_match = False

            for child in curr.children:
                prefix_len = self._ids_prefix_match(unmatched_ids, child.ids)

                if len(child.ids) == prefix_len:
                    match_end_idx += prefix_len
                    child.count -= 1

                    if child.count == 0:
                        curr.children.remove(child)

                    curr = child
                    has_full_match = True
                    break

            if has_full_match is False:
                raise ValueError("Cannot find full match during deletion traversal!")

    
if __name__ == "__main__":
    # Create a test instance
    tree = RadixTree()
    
    # Test case 1: Insert sequences
    print("Test Case 1: Inserting sequences")
    test_sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4],
        [1, 3, 4]
    ]
    
    for seq in test_sequences:
        print(f"\nInserting sequence: {seq}")
        tree.insert(seq)
        print("\nTree structure after insertion:")
        tree.pretty_print()
        
    prefix_match_test_seq = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 7],
        [1, 2, 8],
        [1, 9],
        [10]
    ]

    print("\nTest Case 2: Prefix matching")
    for seq in prefix_match_test_seq:
        print(f"\nPrefix matching sequence: {seq}")
        prefix = tree.prefix_match(seq)
        print(f"Matched prefix: {prefix}")

    # Test case 3: Deletion
    print("\nTest Case 3: Deletion")
    test_sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4],
        [1, 3, 4]
    ]

    for seq in test_sequences:
        print(f"\nDeleting sequence: {seq}")
        tree.delete(seq)
        print("\nTree structure after deletion:")
        tree.pretty_print()