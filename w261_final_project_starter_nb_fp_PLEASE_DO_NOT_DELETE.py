# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Reading/Writing to Shared Data
# MAGIC

# COMMAND ----------

## Place this cell in any team notebook that needs access to the team cloud storage.


# The following blob storage is accessible to team members only (read and write)
# access key is valid til TTL
# after that you will need to create a new SAS key and authenticate access again via DataBrick command line
secret_scope = "261-fp-scope"
secret_key   = "261-fp-scope-key"    
blob_container  = "dmcdonald-261-fp-container"       # The name of your container created in https://portal.azure.com
storage_account = "dmcdonald" # The name of your Storage account created in https://portal.azure.com
team_blob_url        = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket

# the 261 course blob storage is mounted here on the DataBricks workspace.
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)
import pandas as pd
pdf = pd.DataFrame([[1, 2, 3, "Jane"], [2, 2,2, None], [12, 12,12, "John"]], columns=["x", "y", "z", "a_string"])
df = spark.createDataFrame(pdf) # Create a Spark dataframe from a pandas DF

# The following can write the dataframe to the team's Cloud Storage  
# Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
df.write.parquet(f"{team_blob_url}/test", mode='overwrite')



# see what's in the blob storage root folder 
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Abstract
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


