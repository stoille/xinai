# xᵢⁿai Research Toolkit

xinai is a lightweight toolkit for AI research, designed to facilitate answering complex questions that require preprocessing large datasets and performing out-of-core training. It offers an end-to-end pipeline for data handling, model training, and interpretability analysis, with a focus on scalability and reproducibility.

## Key Features

- Scalable data preprocessing using Apache Spark
- Distributed model training with Horovod
- Integration with MLflow for experiment tracking and reproducibility
- Interpretability analysis tools using Captum
- Support for out-of-core training for large datasets and models
- Comprehensive testing suite for ensuring reliability

This toolkit is particularly suited for researchers and data scientists working on questions such as:
- "How does model drift impact attention and response variance in transformer architectures?"
- "What are the effects of different preprocessing techniques on model interpretability?"
- "How do attention patterns in large language models evolve during fine-tuning on domain-specific tasks?"

By providing a robust framework for handling large-scale data and models, this toolkit aims to accelerate research in AI interpretability and promote more transparent and understandable AI systems.

## Project Structure

```
xinai/
├── MLproject
├── conda.yaml
├── CONTRIBUTING.md
├── LICENSE.md
├── README.md
├── data/
│   └── raw/
│       └── sample_data.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── interpretability_analysis.py
│   ├── setup_spark.py
│   ├── setup_horovod.py
│   └── utils.py
├── config/
│   └── model_config.yaml
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
├── models/
│   └── .gitkeep
├── scripts/
│   ├── start-spark-master.sh
│   ├── start-spark-worker.sh
│   └── stop-spark-cluster.sh
└── .gitignore
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/stoille/xinai.git
   cd xinai
   ```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/stoille/xinai.git
   cd xinai
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f conda.yaml
   conda activate xinai_env
   ```

3. Install the project in editable mode:
   ```
   pip install -e .
   ```

## Cluster Setup

### Spark Cluster

1. Install Spark on all nodes. Download from the [Apache Spark website](https://spark.apache.org/downloads.html).

2. Configure the Spark master:
   ```
   ./sbin/start-master.sh
   ```

3. Configure Spark workers and connect them to the master:
   ```
   ./sbin/start-slave.sh spark://master:7077
   ```

4. Set the following environment variables:
   ```
   export SPARK_MASTER_HOST=<master-ip>
   export SPARK_MASTER_PORT=7077
   ```

### Horovod Cluster

1. Ensure OpenMPI or another MPI implementation is installed on all nodes.

2. Install Horovod with CUDA support:
   ```
   HOROVOD_GPU_OPERATIONS=CUDA pip install horovod[pytorch]
   ```

3. Test Horovod installation:
   ```
   horovodrun -np 4 python -c "import horovod.torch as hvd; hvd.init()"
   ```

### MLflow Server

1. Install MLflow:
   ```
   pip install mlflow
   ```

2. Start the MLflow server:
   ```
   mlflow server --host 0.0.0.0 --port 5000
   ```

3. Set the MLflow tracking URI:
   ```
   export MLFLOW_TRACKING_URI=http://<mlflow-server-ip>:5000
   ```

## Usage

### Data Preprocessing

To preprocess the data:

```
mlflow run . -e preprocess
```

### Model Training

To train the model:

```
mlflow run . -e train
```

### Model Evaluation

To evaluate the model:

```
mlflow run . -e evaluate
```

### Interpretability Analysis

To run the interpretability analysis:

```
mlflow run . -e analyze
```

## Running Tests

To run the tests:

```
python -m unittest discover tests
```

## Exploratory Analysis

You can find an exploratory Jupyter notebook in the `notebooks/` directory. To run it:

1. Start Jupyter:
   ```
   jupyter notebook
   ```

2. Navigate to `notebooks/exploratory_analysis.ipynb` and open it.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the BSD-3 License - see the [LICENSE.md](LICENSE.md) file for details.
