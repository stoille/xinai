import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF

def preprocess_data(spark, data_path):
    # Load data
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Tokenization
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    words_data = tokenizer.transform(df)
    
    # Remove stop words
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    filtered_data = remover.transform(words_data)
    
    # TF-IDF
    cv = CountVectorizer(inputCol="filtered", outputCol="tf")
    cv_model = cv.fit(filtered_data)
    tf_data = cv_model.transform(filtered_data)
    
    idf = IDF(inputCol="tf", outputCol="features")
    idf_model = idf.fit(tf_data)
    tfidf_data = idf_model.transform(tf_data)
    
    return tfidf_data

def main(args):
    spark = SparkSession.builder.appName("xᵢⁿai - Data Preprocessing").getOrCreate()
    
    preprocessed_data = preprocess_data(spark, args.data_path)
    
    # Save preprocessed data
    preprocessed_data.write.parquet("data/processed/processed_data.parquet")
    
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for xᵢⁿai")
    parser.add_argument("--data_path", type=str, default="data/raw/sample_data.csv", help="Path to the raw data")
    args = parser.parse_args()
    main(args)
