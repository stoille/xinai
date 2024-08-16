import unittest
import torch
from src.model_training import prepare_data

class TestModelTraining(unittest.TestCase):
    def test_prepare_data(self):
        # Mock data and tokenizer
        class MockTokenizer:
            def __call__(self, texts, **kwargs):
                return {
                    'input_ids': [[1, 2, 3], [4, 5, 6]],
                    'attention_mask': [[1, 1, 1], [1, 1, 1]]
                }

        data = {'text': ['Test sentence 1', 'Test sentence 2'], 'label': [0, 1]}
        tokenizer = MockTokenizer()

        # Prepare data
        dataset = prepare_data(data, tokenizer, max_length=10)

        # Check if the dataset has the correct length
        self.assertEqual(len(dataset), 2)

        # Check if the dataset returns tensors
        self.assertIsInstance(dataset[0][0], torch.Tensor)
        self.assertIsInstance(dataset[0][1], torch.Tensor)
        self.assertIsInstance(dataset[0][2], torch.Tensor)

if __name__ == '__main__':
    unittest.main()
