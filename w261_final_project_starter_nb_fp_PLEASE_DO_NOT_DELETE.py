# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
print("Welcome to the W261 final project John!!") #hey everyone!Hi HELLO good morning


# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

dbutils.fs.help()

# COMMAND ----------

data_BASE_DIR = "/mnt/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# Inspect the Mount's Final Project folder 
# Please IGNORE dbutils.fs.cp("/mnt/mids-w261/datasets_final_project/stations_data/", "/mnt/mids-w261/datasets_final_project_2022/stations_data/", recurse=True)
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

display(dbutils.fs.ls(f"{data_BASE_DIR}stations_data/"))

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_flights = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
#dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/
display(df_flights)

# COMMAND ----------

#500,000 flights per month; 1,500,000 per quarter
# we got double that amount. Why? We have uplicate flights
print(f"{df_flights.count():,} flights; {len(df_flights.columns)} columns")  

# COMMAND ----------

#Calculate how many nulls are in each column  
from pyspark.sql.functions import col,isnan,when,count
display(df_flights.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_flights.columns]
   ).toPandas())


# COMMAND ----------

#Calculate how many nulls are in each column  
from pyspark.sql.functions import col,isnan,when,count
df_flights.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_flights.columns]
   ).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') < "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

# Load the 2015 Q1 for Weather
df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/").filter(col('DATE') > "2015-04-01T00:00:00.000")
display(df_weather)

# COMMAND ----------

print(f"{df_weather.count():,} rows, {len(df_weather.columns)} columns")

# COMMAND ----------

df_stations = spark.read.parquet(f"{data_BASE_DIR}stations_data/*")
display(df_stations)
print(f"{df_stations.count():,} rows; {len(df_stations.columns)} columns")

# COMMAND ----------

df_stations.createOrReplaceTempView("stations")
df_stations.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC Abstract [ADD HERE]

# COMMAND ----------


