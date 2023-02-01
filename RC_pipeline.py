###########################################################################
# * Authors: Valèria Caro & Marina Grifell
# * Date: 15/01/2023
# * Title: Run-time Classifier pipeline (Pipeline 3)
# * Objective: The objective of this last pipeline is, once the model has
# been created, and given a new record, predict if the aircraft is going to
# go for unscheduled maintenance.
##########################################################################


##########################################################################
# IMPORTS SECTION
##########################################################################

import os
import sys
import csv
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession, Row
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

# * Function name: read_input()
# * Receives: -
# * Returns: df with input information (new_record)
# * Objective: asks the user to enter pairs of aircraf and time
# and creates a data frame with the user information

def read_input():
    # Creates a dataframe with the inputs
    new_input = []
    print("Enter your data until you are done, then enter exit.")
    while True:
        time = input("Introduce the timeId you want to predict (YYYY-MM-DD): ")
        if time == "exit":
            print("Computing your prediction(s)...")
            break
        # Convert the timestamp string to a datetime object
        timeid = datetime.strptime(time, '%Y-%m-%d')
        aircraftid = input("Introduce the aircraftId you want to predict: ")
        new_input.append((timeid, aircraftid))
    new_record = spark.createDataFrame(new_input, ["timeid", "aircraftid"])

    return new_record

# * Function name: obtain_table_DW()
# * Receives: -
# * Returns: df with DW information (table_DW)
# * Objective: connects to DW database and extracts its data: aircraftId, timeId,
# flighthours, flightcycles, delayedminutes

def obtain_table_DW():
    # Creates a table from the table aircraftutilization that contains the KPI's needed from the database DW
    table_DW = (spark.read
                           .format("jdbc")
                           .option("driver", "org.postgresql.Driver")
                           .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                           .option("dbtable", "public.aircraftutilization")
                           .option("user", "marina.grifell")
                           .option("password", "DB010902")
                           .load())
    # Selects only the needed attributes in order to be more efficient
    return table_DW.select("timeid", "aircraftid", "flighthours", "flightcycles", "delayedminutes")

# * Function name: obtain_table_CSV()
# * Receives: new_record
# * Returns: df with csvs information about pairs in new_record (table_CSV)
# * Objective: reads a set of csvs and extracts their data: aircraftId, timeId,
# sensor values average if they are in the requested list of the user (new_record)

def obtain_prepared_csvs(new_record):
    # Prepares the schema for csv files data, identified by timeid and aircraftid with their corresponding average measurements of the sensors
    CSV = StructType([
          StructField("date", TimestampType(), True),
          StructField("series", StringType(), True),
          StructField("value", DoubleType(), True)
          ])

    files = [f for f in os.listdir('/Users/marina/Desktop/DADES/BDA/projecte_BD/CodeSkeleton/resources/trainingData') if f.endswith('.csv')]

    df_csv = []
    for file in files:
       df = spark.read.csv('/Users/marina/Desktop/DADES/BDA/projecte_BD/CodeSkeleton/resources/trainingData/' + file, header=True, inferSchema=True, sep = ';', schema = CSV)
       # Gets the timeid and the value for that timeid
       df = df.select("date", "value").withColumn("date", substring(df.date, 0, 10)).withColumnRenamed("date", "timeid_csv")
       # Gets the aircraft id from the cvs file
       df = df.withColumn ("aircraftid_csv", lit(str(file)[-10:-4]))
       df_csv.append(df)

    # Generates a df with all the rows of all the cvs files with the aircrafid and timeid given above
    df_csvs = df_csv[0]
    for df in df_csv[1:]:
        df_csvs = df_csvs.union(df)

    join_csvs_input = df_csvs.join(new_record, [new_record.aircraftid == df_csvs.aircraftid_csv, new_record.timeid == df_csvs.timeid_csv]).drop("aircraftid", "timeid")
    # Uses the generated df to gather all the rows with the same aircraftid and timeid (could be from more than one  cvs file) and generates their average
    table_CSV = join_csvs_input.groupBy("timeid_csv", "aircraftid_csv").agg(avg("value")).withColumnRenamed("avg(value)", "average")

    return table_CSV

# * Function name: obtain_matrix_new_record()
# * Receives: new_record
# * Returns: df with average and KPIs for the pairs requested (matrix_new_record)
# * Objective: gets KPIs and average sensor value for the aicrafts and time
# asked for the user stored in new_record and constructs a new matrix

def obtain_matrix_new_record(new_record):

    table_CSV = obtain_prepared_csvs(new_record)
    table_DW = obtain_table_DW()
    matrix_new_record = table_CSV.join(table_DW, [table_DW.aircraftid == table_CSV.aircraftid_csv, table_DW.timeid == table_CSV.timeid_csv]).drop("aircraftid_csv", "timeid_csv", "aircraftid", "timeid")
    return matrix_new_record

# * Function name: record_classifier()
# * Receives: matrix_new_record, model
# * Returns: predictions list
# * Objective: uses the model to label features in matrix_new_record

def record_classifier(matrix_new_record, model):
    # Vectorizes data as features
    assembler = VectorAssembler(inputCols=(["flighthours", "flightcycles", "delayedminutes", "average"]), outputCol=("features"))
    output = assembler.transform(matrix_new_record)
    output = output.select("features")
    # Gets the prediction
    predictions = model.transform(output)
    return predictions.select("prediction").collect()

# * Function name: output_prediction()
# * Receives: predictions, new_record
# * Returns: -
# * Objective: tells the user the prediction for each aicraft and time:
    #   If value 0.0 - No maintenance
    #   Else - Scheduled maintenance

def output_prediction(predictions, new_record):
    pairs = [(row.timeid, row.aircraftid) for row in new_record.collect()]
    for i in range (len(pairs)):
        # Convert the datetime object to a string in the "YYYY-MM-DD" format
        timeid_str = pairs[i][0].strftime("%Y-%m-%d")
        print("Your aircraft " + pairs[i][1] + " on " + timeid_str + " will have", end=' ')
        if predictions[i].prediction == 0: print("no maintenance.")
        else: print("unscheduled maintennace.")

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

    # Executes pipeline
    new_record = read_input()
    matrix_new_record = obtain_matrix_new_record(new_record)
    # Loads the validated model
    model = PipelineModel.load("model")
    predictions = record_classifier(matrix_new_record, model)
    output_prediction(predictions, new_record)
