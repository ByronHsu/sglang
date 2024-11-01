from sglang.srt.router.radix_tree import RadixTree
import unittest

class TestRadixTree(unittest.TestCase):
    def setUp(self):
        self.tree = RadixTree()

    def test_insert_single_sequence(self):
        """Test inserting a single sequence"""
        sequence = [1, 2, 3, 4]
        self.tree.insert(sequence)
        self.assertEqual(self.tree.root.count, 1)
        self.assertEqual(len(self.tree.root.children), 1)
        self.assertEqual(self.tree.root.children[0].ids, sequence)

    def test_insert_shared_prefix(self):
        """Test inserting sequences with shared prefixes"""
        seq1 = [1, 2, 3, 4]
        seq2 = [1, 2, 3, 5]
        
        self.tree.insert(seq1)
        self.tree.insert(seq2)

        # Root should have count 2
        self.assertEqual(self.tree.root.count, 2)
        
        # Check the shared prefix [1, 2, 3]
        first_child = self.tree.root.children[0]
        self.assertEqual(first_child.ids, [1, 2, 3])
        self.assertEqual(first_child.count, 2)
        
        # Check the split nodes [4] and [5]
        self.assertEqual(len(first_child.children), 2)
        self.assertEqual(first_child.children[0].ids, [4])
        self.assertEqual(first_child.children[1].ids, [5])
        self.assertEqual(first_child.children[0].count, 1)
        self.assertEqual(first_child.children[1].count, 1)

    def test_prefix_match_exact(self):
        """Test prefix matching with exact matches"""
        sequence = [1, 2, 3, 4]
        self.tree.insert(sequence)
        
        # Test exact match
        match = self.tree.prefix_match(sequence)
        self.assertEqual(match, sequence)

    def test_prefix_match_partial(self):
        """Test prefix matching with partial matches"""
        self.tree.insert([1, 2, 3, 4])
        
        # Test partial matches
        test_cases = [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4]),  # Longer sequence
            ([1, 2, 3], [1, 2, 3]),           # Shorter sequence
            ([1, 2, 5], [1, 2]),              # Different branch
            ([2, 3, 4], []),                  # No match
        ]
        
        for test_input, expected in test_cases:
            with self.subTest(test_input=test_input):
                match = self.tree.prefix_match(test_input)
                self.assertEqual(match, expected)

    def test_delete_sequence(self):
        """Test deleting sequences"""
        sequences = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4]
        ]
        
        # Insert sequences
        for seq in sequences:
            self.tree.insert(seq)
            
        # Delete first sequence
        self.tree.delete(sequences[0])
        
        # Check counts are updated
        self.assertEqual(self.tree.root.count, 2)
        
        # Try to delete non-existent sequence
        with self.assertRaises(ValueError):
            self.tree.delete([1, 2, 3, 6])

    def test_delete_with_node_removal(self):
        """Test that nodes are properly removed when count reaches 0"""
        sequence = [1, 2, 3, 4]
        self.tree.insert(sequence)
        self.tree.delete(sequence)
        
        # Root should be empty after deletion
        self.assertEqual(self.tree.root.count, 0)
        self.assertEqual(len(self.tree.root.children), 0)

    def test_complex_operations(self):
        """Test a complex sequence of operations"""
        operations = [
            ([1, 2, 3], "insert"),
            ([1, 2, 3, 4], "insert"),
            ([1, 2], "insert"),
            ([1, 2, 3], "delete"),
            ([1, 2, 4], "insert"),
        ]
        
        for sequence, operation in operations:
            if operation == "insert":
                self.tree.insert(sequence)
            else:
                self.tree.delete(sequence)
        
        # Verify final state
        self.assertEqual(self.tree.root.count, 3)
        self.assertTrue(self.tree.prefix_match([1, 2, 3, 4]) == [1, 2, 3, 4])
        self.assertTrue(self.tree.prefix_match([1, 2, 4]) == [1, 2, 4])

    def test_empty_sequence(self):
        """Test handling of empty sequences"""
        # Insert empty sequence
        self.tree.insert([])
        self.assertEqual(self.tree.root.count, 1)
        
        # Match empty sequence
        match = self.tree.prefix_match([])
        self.assertEqual(match, [])
        
        # Delete empty sequence
        self.tree.delete([])
        self.assertEqual(self.tree.root.count, 0)

if __name__ == "__main__":
    unittest.main()