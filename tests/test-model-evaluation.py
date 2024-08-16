import unittest
import torch
from src.model_evaluation import evaluate

class TestModelEvaluation(unittest.TestCase):
    def test_evaluate(self):
        # Mock model and data loader
        class MockModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return torch.nn.functional.softmax(torch.randn(input_ids.size(0), 2), dim=1)

        class MockDataLoader:
            def __init__(self):
                self.data = [
                    (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]), torch.tensor([0])),
                    (torch.tensor([[4, 5, 6]]), torch.tensor([[1, 1, 1]]), torch.tensor([1]))
                ]

            def __iter__(self):
                return iter(self.data)

        model = MockModel()
        test_loader = MockDataLoader()

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate(model, test_loader)

        # Check if the metrics are valid
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)

if __name__ == '__main__':
    unittest.main()
