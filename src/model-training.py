import argparse
import yaml
import torch
import horovod.torch as hvd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql import SparkSession

def load_data(spark, data_path):
    df = spark.read.parquet(data_path)
    return df.toPandas()

def prepare_data(data, tokenizer, max_length):
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']), torch.tensor(data['label'].tolist()))
    return dataset

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, mask, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data, attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main(args):
    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # Load configuration
    with open(args.model_config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Spark
    spark = SparkSession.builder.appName("xᵢⁿai - Model Training").getOrCreate()

    # Load and prepare data
    data = load_data(spark, args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    dataset = prepare_data(data, tokenizer, config['max_length'])
    
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_sampler)

    # Build model
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=config['num_labels'])
    model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'] * hvd.size())

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Train the model
    for epoch in range(1, config['epochs'] + 1):
        train(model, train_loader, optimizer, epoch)

    # Save the model (only on the first worker)
    if hvd.rank() == 0:
        torch.save(model.state_dict(), "models/trained_model.pth")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training for xᵢⁿai")
    parser.add_argument("--data_path", type=str, default="data/processed/processed_data.parquet", help="Path to the processed data")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml", help="Path to the model configuration file")
    args = parser.parse_args()
    main(args)
