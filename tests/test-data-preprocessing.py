import unittest
from pyspark.sql import SparkSession
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestDataPreprocessing").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_preprocess_data(self):
        # Create a sample dataframe
        data = [("This is a test sentence.", 1),
                ("Another test sentence.", 0)]
        df = self.spark.createDataFrame(data, ["text", "label"])

        # Preprocess the data
        result = preprocess_data(self.spark, df)

        # Check if the result has the expected columns
        self.assertIn("features", result.columns)
        self.assertIn("label", result.columns)

        # Check if the number of rows is preserved
        self.assertEqual(result.count(), 2)

if __name__ == '__main__':
    unittest.main()
