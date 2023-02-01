import os
import sys
import csv
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import *


# Path of the .jar of postgresql
HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "/Users/marina/Desktop/DADES/3R/1rquadri/BDA/LAB/Projecte_BD/Training/TrainingCode/resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.9"
PYSPARK_DRIVER_PYTHON = "python3.9"


def obtain_table_DW():
    # Creates a table with the table aircraftutilization that contains the KPI's needed from the database DW
    table_DW = (spark.read
                           .format("jdbc")
                           .option("driver", "org.postgresql.Driver")
                           .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                           .option("dbtable", "public.aircraftutilization")
                           .option("user", "marina.grifell")
                           .option("password", "DB010902")
                           .load())
    return table_DW.select("timeid", "aircraftid", "flighthours", "flightcycles", "delayedminutes")



def obtain_table_AMOS():
    # Creates a table with the table operationinterruption (oldinstance) that contains the kind of interruption from the database AMOS
    table_AMOS = (spark.read
                             .format("jdbc")
                             .option("driver", "org.postgresql.Driver")
                             .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
                             .option("dbtable", "oldinstance.operationinterruption")
                             .option("user", "marina.grifell")
                             .option("password", "DB010902")
                             .load())
    return table_AMOS.filter(table_AMOS.subsystem == "3453").select("departure", "aicraftregistration", "kind")



def obtain_table_CSV():
    CSV = StructType([
          StructField("date", TimestampType(), True),
          StructField("series", StringType(), True),
          StructField("value", DoubleType(), True)
          ])

    files = [f for f in os.listdir('/Users/marina/Desktop/DADES/3R/1rquadri/BDA/LAB/Projecte_BD/delveryTemplates/CodeSkeleton/resources/trainingData') if f.endswith('.csv')]

    df_csv = []
    for file in files:
       sum = 0
       total_rows = 0
       df = spark.read.csv('/Users/marina/Desktop/DADES/3R/1rquadri/BDA/LAB/Projecte_BD/delveryTemplates/CodeSkeleton/resources/trainingData/' + file, header=True, inferSchema=True, sep = ';', schema = CSV)
       df = df.select("date", "value").withColumn("date", substring(df.date, 0, 10)).withColumnRenamed("date", "timeid_csv")
       df = df.withColumn ("aircraftid_csv", lit(str(file)[-10:-4]))
       df_csv.append(df)

    df_csvs = df_csv[0]
    for df in df_csv[1:]:
        df_csvs = df_csvs.union(df)

    df_csvs = df_csvs.groupBy("timeid_csv", "aircraftid_csv").agg(avg("value")).withColumnRenamed("avg(value)", "average")

    return df_csvs


def obtain_matrix(table_CSV, table_DW, table_AMOS):
    # join per aircraft i timeid de taula del csv i taula DW
    # inner join by default
    df_join = table_CSV.join(table_DW, [table_DW.aircraftid == table_CSV.aircraftid_csv, table_DW.timeid == table_CSV.timeid_csv]).drop("aircraftid_csv", "timeid_csv")
#    df_join = df_join.select(df_join.timeid, df_join.aircraftid, df_join.average, df_join.flighthours, df_join.flightcycles, df_join.delayedminutes)
    primary_keys = table_AMOS.select("departure", "aircraftregistration").rdd.map(lambda row: (row['departure'], row['aircraftregistration'])).collect()
    matrix = df_join.withColumn("maintenance", when((col('timeid').isin([pair[0] for pair in primary_keys]) & col('aircraftid').isin([pair[1] for pair in primary_keys])), "unscheduel maintenance").otherwise("no maintenance"))
    #FER ELS 7 DIES PREVIS
    return matrix


def DM_pipeline():
    table_DW = obtain_table_DW()
    table_AMOS = obtain_table_AMOS()
    # seleccionem nomes els 3453 de sensors perqupe son els unics que analitzem
    table_CSV = obtain_table_CSV()
    matrix = obtain_matrix(table_CSV, table_DW, table_AMOS)
    with open('/Users/marina/Desktop/matrix.csv', 'w', newline = '') as csv_output:
        writer = csv.writer(csv_output)
        writer.writerows(matrix)


if(__name__ == "__main__"):

    # Creates the configuration
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    # Builds the Spark session
    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    # Data Management Pipeline
    DM_pipeline()

    # Data Analysis Pipeline
    # Vectorizes the dataframe

    assembler = VectorAssembler(inputCols=(["flighthours", "flightcycles", "delayedminutes", "average"]), outputCol=("features"))
    output = assembler.transform(matrix)
    output = output.select("features", "maintenance")

    labelIndexer = StringIndexer(inputCol="maintenance", outputCol="indexedLabel").fit(output)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(output)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = output.randomSplit([0.7, 0.3])
    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    # Make predictions.
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))
    treeModel = model.stages[2]
    # summary only
    print(treeModel)





    #Index labels, adding metadata to the label column.
    #Fit on whole dataset to include all labels in index.

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.



