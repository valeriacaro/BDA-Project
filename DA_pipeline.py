###########################################################################
# * Authors: Valèria Caro & Marina Grifell
# * Date: 15/01/2023
# * Title: Data Analysis pipeline (Pipeline 2)
# * Objective: The objective of this second pipeline is to get a trained a
# Decision Tree classifier by using the data matrix created in the previous
# pipeline.
##########################################################################

##########################################################################
# IMPORTS SECTION
##########################################################################

import os
import sys
import csv
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import *
from pyspark.sql import DataFrameWriter
from datetime import datetime, timedelta


##########################################################################
# PATHS SECTION
##########################################################################


# Path of the .jar of postgresql
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "/Users/marina/Desktop/DADES/BDA/projecte_BD/CodeSkeleton/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.9"
PYSPARK_DRIVER_PYTHON = "python3.9"

##########################################################################
# FUNCTIONS SECTION
##########################################################################

# * Function name: obtain_matrix()
# * Receives: -
# * Returns: df with avg, KPIs and label information (matrix)
# * Objective: reads the csv generated on pipeline 1 and extracts its data

def obtain_matrix():
    # Gets the matrix from the csv previously saved
    CSV = StructType([
          StructField("average", DoubleType(), True),
          StructField("flighthours", DoubleType(), True),
          StructField("flightcycles", DoubleType(), True),
          StructField("delayedminutes", DoubleType(), True),
          StructField("maintenance", StringType(), True)
          ])

    matrix = spark.read.csv('/Users/marina/Desktop/DADES/BDA/projecte_BD/matrix', header=True, schema = CSV)
    return matrix

# * Function name: vectorize_dataframe()
# * Receives: matrix
# * Returns: vectorized matrix (df_vec)
# * Objective: vectorizes data features to get data prepared to train the model

def vectorize_dataframe(matrix):
    # Vectorizes features for the model
    assembler = VectorAssembler(inputCols=(["flighthours", "flightcycles", "delayedminutes", "average"]), outputCol=("features"))
    output = assembler.transform(matrix)
    # Gets a vector with features and label (maintenance)
    df_vec = output.select("features", "maintenance")
    return df_vec

# * Function name: data_preparation()
# * Receives: df_vec
# * Returns: two sets - training and validation data
# * Objective: separes data in two sets: training and validation data.
# The first set will contain 70% of data, while the second one the rest of it.

def data_preparation(df_vec):
    # Splits the data into training and validation datasets (30% held out for testing)
    (trainingData, validationData) = df_vec.randomSplit([0.7, 0.3])
    return trainingData, validationData

# * Function name: model_training()
# * Receives: trainingData, labelIndexer, featureIndexer
# * Returns: a trained Decision Tree Classifier model (model)
# * Objective: trains a Decision Tree Classifier with data on training set

def model_training(trainingData, labelIndexer, featureIndexer):
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    model = pipeline.fit(trainingData)
    return model

# * Function name: model_validation()
# * Receives: validationData, model
# * Returns: -
# * Objective: validates the model already trained with data in validation
# test and computes the test error, the model's accuracy and recall.

def model_validation(validationData, model):
    # Makes predictions according to the validation data
    predictions = model.transform(validationData)
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features")
    # Select (prediction, true label) and compute test error
    evaluator_1 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_1.evaluate(predictions)
    # Prints traditional evaluation metrics such as Test error and Accuracy
    print("Test Error = %g " % (1.0 - accuracy))
    print("Accuracy = %g " % (accuracy))
    treeModel = model.stages[2]
    # Selects (prediction, true label) and computes recall
    evaluator_2 = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="recallByLabel")
    recall = evaluator_2.evaluate(predictions)
    # Prints traditional evaluation metrics such as Recall
    print("Recall = %g " % (recall))

##########################################################################
# MAIN SECTION
##########################################################################

if(__name__ == "__main__"):

    # Creates the configuration
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    # Builds the Spark session
    spark = SparkSession.builder.config(conf=conf).master("local").appName("Training").getOrCreate()
    sc = pyspark.SparkContext.getOrCreate()

    # Executes the pipeline
    matrix = obtain_matrix()
    df_vec = vectorize_dataframe(matrix)
    # Indexes labels, adding metadata to the label column.
    labelIndexer = StringIndexer(inputCol="maintenance", outputCol="indexedLabel").fit(df_vec)
    # Automatically identifies categorical features, and indexes them.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df_vec)
    [trainingData, validationData] = data_preparation(df_vec)
    model = model_training(trainingData, labelIndexer, featureIndexer)
    model_validation(validationData, model)
    #Stores the validation model in a folder
    model.write().overwrite().save("model")
