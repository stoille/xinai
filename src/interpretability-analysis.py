import argparse
import torch
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

def load_data(spark, data_path):
    df = spark.read.parquet(data_path)
    return df.toPandas()

def prepare_data(data, tokenizer, max_length):
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(data['label'].tolist()))
    return dataset

def analyze_interpretability(model, input_ids, attention_mask, target):
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_ids, target=target, n_steps=200, return_convergence_delta=True)
    
    lc = LayerConductance(model, model.bert.encoder.layer[-1])
    layer_attrs = lc.attribute(input_ids, target=target)
    
    nc = NeuronConductance(model, model.bert.encoder.layer[-1])
    neuron_attrs = nc.attribute(input_ids, target=target)
    
    return attributions, layer_attrs, neuron_attrs

def visualize_attributions(tokenizer, input_ids, attributions):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    vis = viz.VisualizationDataRecord(
        attributions[0].sum(dim=-1).cpu().numpy(),
        pred_prob=0.0,
        pred_class='',
        true_class='',
        attr_class='',
        attr_score=attributions[0].sum().cpu().item(),
        raw_input=tokens,
        convergence_score=0.0
    )
    viz.visualize_text([vis])

def main(args):
    # Initialize Spark
    spark = SparkSession.builder.appName("xᵢⁿai - Analysis").getOrCreate()

    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    # Load and prepare data
    data = load_data(spark, args.data_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = prepare_data(data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=1)

    # Analyze interpretability
    for input_ids, attention_mask, target in dataloader:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        target = target.cuda()

        attributions, layer_attrs, neuron_attrs = analyze_interpretability(model, input_ids, attention_mask, target)

        # Visualize attributions
        visualize_attributions(tokenizer, input_ids, attributions)

        # Log results with MLflow
        with mlflow.start_run():
            mlflow.log_metric("total_attribution", attributions.sum().item())
            mlflow.log_metric("layer_attribution", layer_attrs.sum().item())
            mlflow.log_metric("neuron_attribution", neuron_attrs.sum().item())

        # Save visualization
        plt.savefig("interpretability_visualization.png")
        mlflow.log_artifact("interpretability_visualization.png")

        break  # For demonstration, we only process the first sample

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretability Analysis for AI")
    parser.add_argument("--model_path", type=str, default="models/trained_model.pth", help="Path to the trained model")
    parser.add_argument("--data_path", type=str, default="data/processed/test_data.parquet", help="Path to the data")
    args = parser.parse_args()
    main(args)
