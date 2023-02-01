###########################################################################
# * Authors: Valèria Caro & Marina Grifell
# * Date: 15/01/2023
# * Title: Data Management pipeline (Pipeline 1)
# * Objective: The objective of this pipeline  is to generate a matrix where
# the rows denote the information of an aircraft per day, and the columns
# refer to the FH, FC and DM KPIs, and the average measurement of the 3453
# sensor.
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

# * Function name: obtain_table_CSV()
# * Receives: -
# * Returns: df with csvs information (table_CSV)
# * Objective: reads a set of csvs and extracts their data: aircraftId, timeId,
# sensor values average

def obtain_table_CSV():
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

    # Uses the generated df to gather all the rows with the same aircraftid and timeid (could be from more than one  cvs file) and generates their average
    df_csvs = df_csvs.groupBy("timeid_csv", "aircraftid_csv").agg(avg("value")).withColumnRenamed("avg(value)", "average")

    return df_csvs

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

# * Function name: obtain_table_AMOS()
# * Receives: -
# * Returns: df with AMOS information about sensor 3453 (table_AMOS)
# * Objective: connects to AMOS database and extracts the aircraft registrations
# that had unscheduled maintenance and when did their start for sensor 3453

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
    # Filters already the particular sensor and unscheduled types of maintenance and selects only the needed attributes in order to be more efficient
    return table_AMOS.filter((table_AMOS.subsystem == "3453") & (table_AMOS.kind.isin(['Delay', 'Safety', 'AircraftOnGround']))).select("starttime", "aircraftregistration").withColumn('starttime', to_date(col('starttime')))

# * Function name: obtain_matrix()
# * Receives: table_CSV, table_DW, table_AMOS
# * Returns: df with avg values, KPIs and maintenance label (matrix)
# * Objective: gets KPIs from table_DW for the aircraftId and timeId in table_CSV and
# looks at table_AMOS to see if in that timeId, or in the following 7 days, the
# aircraft has an unscheduled maintenance; if so, labels it as unscheduled maintenance,
# otherwise labels it as no maintenance

def obtain_matrix(table_CSV, table_DW, table_AMOS):
    # Joins the previous CSV and DW tables
    df_join = table_CSV.join(table_DW, [table_DW.aircraftid == table_CSV.aircraftid_csv, table_DW.timeid == table_CSV.timeid_csv]).drop("aircraftid_csv", "timeid_csv")
    # Saves all the pairs of (timeid, aircraftid) contained in the AMOS table
    primary_keys = table_AMOS.select("starttime", "aircraftregistration").rdd.map(lambda row: (row['starttime'], row['aircraftregistration'])).collect()
    # Creates the column maintenance by searching every timeid and aircraftid from the df_join in the AMOS pairs,
    # if it is found in the table, it is labeled as unsecheduled maintenance, otherwise as no maintenance.
    # Not only it looks for the current timeid, it also checks the next 7 days in order to be able to make a prediction later on.
    matrix = df_join.withColumn("maintenance", when((col('timeid').isin([pair[0]+ timedelta(days=i) for i in range (0,8) for pair in primary_keys]) & col('aircraftid').isin([pair[1] for pair in primary_keys])), "unscheduled maintenance").otherwise("no maintenance"))
    # Aicraft name and time id are not needed anymore
    matrix = matrix.drop("aircraftid", "timeid")

    return matrix

# * Function name: save_matrix_to_csv()
# * Receives: matrix
# * Returns: -
# * Objective: saves matrix in csv format

def save_matrix_to_csv(matrix):
    matrix.write.mode("overwrite").csv("matrix", header=True)


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
    table_CSV = obtain_table_CSV()
    table_DW = obtain_table_DW()
    table_AMOS = obtain_table_AMOS()
    matrix = obtain_matrix(table_CSV, table_DW, table_AMOS)
    save_matrix_to_csv(matrix)
