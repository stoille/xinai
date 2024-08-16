import argparse
import torch
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data(spark, data_path):
    df = spark.read.parquet(data_path)
    return df.toPandas()

def prepare_data(data, tokenizer, max_length):
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(data['label'].tolist()))
    return dataset

def evaluate(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, mask, target in test_loader:
            output = model(data, attention_mask=mask).logits
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    return accuracy, precision, recall, f1

def main(args):
    # Initialize Spark
    spark = SparkSession.builder.appName("xᵢⁿai - Model Evaluation").getOrCreate()

    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    # Load and prepare test data
    test_data = load_data(spark, args.test_data_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_dataset = prepare_data(test_data, tokenizer, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate(model, test_loader)

    # Log metrics with MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation for xᵢⁿai")
    parser.add_argument("--model_path", type=str, default="models/trained_model.pth", help="Path to the trained model")
    parser.add_argument("--test_data_path", type=str, default="data/processed/test_data.parquet", help="Path to the test data")
    args = parser.parse_args()
    main(args)
