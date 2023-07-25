# Databricks notebook source
# MAGIC %md
# MAGIC # Phase Leader Plan
# MAGIC
# MAGIC |Phase	| Description |	Leader	| Due Date |
# MAGIC | ------|-------------|---------|----------|
# MAGIC |1	| describe datasets, joins, tasks, and metrics	| Diego	 | 07/17|
# MAGIC |2	| eda + base pipeline	| Rithvik	| 07/23 |
# MAGIC |3	| feature engineering + hyperparameter tuning |	Aaron	 | 07/30 |
# MAGIC |4	| model architecture + loss functions + optimal algorithm + final report | 	Adam	| 08/06 |
# MAGIC |5	| presentation	| Saket	| 08/10 |

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 1 Credit
# MAGIC
# MAGIC | Name	| Task |
# MAGIC |-------|------| 
# MAGIC | Diego	| Abstract + block diagram + submit + project title |
# MAGIC | Aaron	| Data Description |
# MAGIC | Rithvik	| Summary visual EDA + Description |
# MAGIC | Adam	| ML Algs + Metrics | 
# MAGIC | Saket	| Summary EDA  |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Abstract
# MAGIC
# MAGIC Creating a model to accurately predict flight delays will transform the airline industry and provide noticeable improvements to the lives of passengers and stakeholders. This project aims to create a model that predicts flight delays based on relevant weather data.
# MAGIC
# MAGIC This project will use two data sources to train our model. We will use the Department of Transportation’s (DoT) TransStats data collection in conjunction with the National Oceanic and Atmospheric Administration’s (NOAA) weather dataset. Both datasets contain data from 2015 to 2021. However, we will primarily be using data up until 2019. These two datasets will be joined by date, allowing us to work using a single dataset.
# MAGIC
# MAGIC Our work begins with exploratory data analysis (see below). 
# MAGIC
# MAGIC a) ARR_DELAY & DEP_DELAY basic stats
# MAGIC b) Arrival and Departure delays by Airline
# MAGIC c) Arrival and Departure delays by City
# MAGIC d) Arrival & Departure delay histogram (log normalised to check for distribution type)
# MAGIC
# MAGIC Additionally, our pipeline is displayed below. Our pipeline begins with data cleaning and processing, a necessary step to handle any null or missing data. We then perform feature selection, removing any features that might be correlated with each other, and then split our data in train/validation/test splits. This leads into our model training and validation stage, where we will iterate on our model hyperparameters to ensure optimal model performance, using AUC scores, precision, recall, etc. Once our model is selected, we will perform model evaluation on the test data set. Finally, (and somewhat theoretically), we will deploy our model to a production environment for others to use. 
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Data
# MAGIC
# MAGIC There are multiple datasets given to us, such as Flights Data collected from the US Department of Transportation that has 31 million + flights with 109 features. We are also provided with the Weather table, extracted from the National Oceanic & Atmospheric Administration repository and has 6 years of data with 630 million datapoints and 177 features. 
# MAGIC We are also given more simple datasets such as Airport metadata (dimensions 18,097 x 10) and Airport codes table which provides the three letter IATA airport code. 
# MAGIC
# MAGIC Exploratory analysis can be performed on this dataset to gain insights. For numerical columns like DEP_DELAY, ARR_DELAY, DEP_DELAY_NEW, and ARR_DELAY_NEW, statistical measures such as mean and standard deviation can be calculated to understand the average delay and its variability. Categorical columns like OP_UNIQUE_CARRIER and CANCELLATION_CODE can be analyzed to determine the distribution of carriers and cancellation reasons. 
# MAGIC
# MAGIC Additionally, the dataset may contain missing or corrupt data. Missing data can be identified by examining null values in the dataset. Outliers, which are extreme values that deviate significantly from the other data points, can be detected using statistical methods such as box plots or by considering predefined thresholds. We will conduct analysis to deal with null values and outliers after carefully studying the context of each feature. For example, for later columns in the Flights Data that contains mostly null values, we can drop the columns if determined to be unnecessary to conduct our final analysis. 
# MAGIC
# MAGIC In summary, this dataset provides detailed information about flights, including dates, carriers, airports, and delays. Exploratory analysis, including statistical measures and identification of missing or corrupt data, can help in understanding patterns and extracting valuable insights.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Exploratory Data Analysis (3 months flight data from 2015 first Quarter)
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
print("Welcome to the W261 final project John!!") #hey everyone!Hi HELLO good morning

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))


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

## We will use this later for EDA

df_weather =  spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data/*")
df_weather = df_weather.filter(df_weather.DATE < "2015-04-01T00:00:00.000").cache()

# COMMAND ----------

df_weather.count()

# COMMAND ----------

# Load 2015 Q1 for Flights        dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/
df_flights = spark.read.parquet(f"{data_BASE_DIR}parquet_airlines_data_3m/")
#dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/
display(df_flights)

# COMMAND ----------

df_flights.select(max('FL_DATE')).show()

# COMMAND ----------

# 500,000 flights per month; 1,500,000 per quarter
# we got double that amount. Why? We have uplicate flights
print(f"{df_flights.count():,} flights; {len(df_flights.columns)} columns")  

# COMMAND ----------

column_list = df_flights.columns
print(column_list)

# COMMAND ----------

from pyspark.sql.functions import mean, stddev, min, max
# Both delayed and early arrivals
df_flights_all = df_flights.select('ARR_DELAY')
display(df_flights_all.select(mean('ARR_DELAY'), stddev('ARR_DELAY'), min('ARR_DELAY'), max('ARR_DELAY')))

## Should we consider absolute values or remove the early arrival data?

# COMMAND ----------

## only the delayed flights -- excluded the early arrivals

df_flights_pos = df_flights.select('ARR_DELAY').filter(df_flights['ARR_DELAY'] > 0)
display(df_flights_pos.select(mean('ARR_DELAY'), stddev('ARR_DELAY'), min('ARR_DELAY'), max('ARR_DELAY')))

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
# Create a histogram of the 'ArrDelay' column
arr_delay = df_flights_pos.toPandas() ## Only has the ARR_DELAY field. So ok to send to the driver
plt.hist(arr_delay['ARR_DELAY'], bins=20)
plt.show()

# COMMAND ----------

df_flights_posarr = df_flights_pos.toPandas()
sns.histplot(df_flights_posarr['ARR_DELAY'], bins=50, log_scale=True)
plt.title('Log Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

df_flights_posdep = df_flights.select('DEP_DELAY').filter(df_flights['DEP_DELAY'] > 0)
display(df_flights_posdep.select(mean('DEP_DELAY'), stddev('DEP_DELAY'), min('DEP_DELAY'), max('DEP_DELAY')))

# COMMAND ----------

## Log normalised Departure Delay histogram

df_flights_posdep = df_flights_posdep.toPandas()

sns.histplot(df_flights_posdep['DEP_DELAY'], bins=50, log_scale=True)
plt.title('Log Distribution of Departure Delays')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

## Checking if there is high correlation between ARR_DELAY and DEP_DELAY
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['ARR_DELAY', 'DEP_DELAY'], outputCol='features')
df_flights_nona = df_flights.select(['ARR_DELAY', 'DEP_DELAY']).na.drop() ## Removing nulls from the two fields
#df_flights_nona.count()
df_vector = assembler.transform(df_flights_nona).select('features')

matrix = Correlation.corr(df_vector, 'features').collect()[0][0]
correlation_matrix = matrix.toArray().tolist()
# print(correlation_matrix)

# Visualize the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, xticklabels=['ARR_DELAY', 'DEP_DELAY'], yticklabels=['ARR_DELAY', 'DEP_DELAY'])
plt.title('Correlation Matrix between ARR_DELAY & DEP_DELAY')
plt.show()

## Seems to show extremely high correlation which makes sense

# COMMAND ----------

## Calculating correlation between ARR_DELAY & Reasons for delay

assembler = VectorAssembler(inputCols=['ARR_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], outputCol='features')
df_flights_res_nona = df_flights.select(['ARR_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']).na.drop() ## Removing nulls from all fields
#df_flights_res_nona.count()
df_flights_vector = assembler.transform(df_flights_res_nona).select('features')

matrix = Correlation.corr(df_flights_vector, 'features').collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

# Visualize the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, xticklabels=['ARR_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], 
            yticklabels=['ARR_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'])
plt.title('Correlation Matrix between ARR_DELAY & Delay Reasons')
plt.show()


# COMMAND ----------

## Calculating correlation between ARR_DELAY & Reasons for delay

assembler = VectorAssembler(inputCols=['DEP_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], outputCol='features')
df_flights_res_nona = df_flights.select(['DEP_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']).na.drop() ## Removing nulls from all fields
#df_flights_res_nona.count()
df_flights_vector = assembler.transform(df_flights_res_nona).select('features')

matrix = Correlation.corr(df_flights_vector, 'features').collect()[0][0]
correlation_matrix = matrix.toArray().tolist()

# Visualize the correlation matrix as a heatmap
sns.heatmap(correlation_matrix, annot=True, xticklabels=['DEP_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], 
            yticklabels=['DEP_DELAY', 'CARRIER_DELAY','WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'])
plt.title('Correlation Matrix between DEP_DELAY & Delay Reasons')
plt.show()

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofmonth

## includes negative arr_delay meaning the early arrivals
## Group the data by year and calculate the average delay time
display(df_flights_nona.groupBy(year('FL_DATE')).agg(mean('ARR_DELAY')))

## Group the data by month and calculate the average delay time
display(df_flights_nona.groupBy(month('FL_DATE')).agg(mean('ARR_DELAY')))

## Group the data by day of month and calculate the average delay time
display(df_flights_nona.groupBy(dayofmonth('FL_DATE')).agg(mean('ARR_DELAY')))

# COMMAND ----------

import plotly.express as px
from pyspark.sql.functions import count

## For the 3 month data here is the flight count by airline
carrier_counts = df_flights.select('OP_UNIQUE_CARRIER').groupBy('OP_UNIQUE_CARRIER').agg(count('*').alias('Flight Count')).toPandas()
carrier_counts = carrier_counts.sort_values(by='Flight Count', ascending=False)
fig = px.bar(carrier_counts, x='OP_UNIQUE_CARRIER', y='Flight Count', title='Number of Flights by Carrier')
fig.update_layout(xaxis_title='Airline Carrier Code')
fig.show()

# COMMAND ----------

## Since this data is only for the 1st quarter of 2015, only looking at day and month
## Additionally looking at all data and not just delays but also early arrivals

df_flights_nonnull = df_flights.na.drop(subset=['FL_DATE','ARR_DELAY', 'DEP_DELAY'])

## Group the data by month and calculate the average delay time
monthly_delay = df_flights_nonnull.groupBy(month('FL_DATE')).agg(mean('ARR_DELAY')).toPandas()
sns.barplot(x='month(FL_DATE)', y='avg(ARR_DELAY)', data=monthly_delay)
plt.title('Average Arrival Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Arrival Delay (minutes)')
plt.show()

## Group the data by day and calculate the average delay time
daily_delay = df_flights_nonnull.groupBy(dayofmonth('FL_DATE')).agg(mean('ARR_DELAY')).toPandas()
sns.barplot(x='dayofmonth(FL_DATE)', y='avg(ARR_DELAY)', data=daily_delay)
plt.title('Average Arrival Delay by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

## Seems that the first few days of the month have a higher likelihood of flight delays

# COMMAND ----------

df_flights_nonnullordest = df_flights.select(['OP_UNIQUE_CARRIER','ARR_DELAY', 'DEP_DELAY','ORIGIN_CITY_NAME', 'DEST_CITY_NAME']).na.drop()

carrier_delay = df_flights_nonnullordest.groupBy('OP_UNIQUE_CARRIER').agg(mean('ARR_DELAY')).toPandas()
sns.barplot(x='OP_UNIQUE_CARRIER', y='avg(ARR_DELAY)', data=carrier_delay)
plt.title('Average Arrival Delay by Airline')
plt.xlabel('Airline')
plt.ylabel('Average Arrival Delay (minutes)')
plt.show()

# Group & Sort by ARR_DELAY and show top 10 descending
origin_delay = df_flights_nonnull.groupBy('ORIGIN_CITY_NAME').agg(mean('ARR_DELAY')).toPandas()
origin_delay = origin_delay.sort_values(by='avg(ARR_DELAY)', ascending=False).head(10)
sns.barplot(x='ORIGIN_CITY_NAME', y='avg(ARR_DELAY)', data=origin_delay)
plt.title('Top 10 Average Arrival Delay by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

# Group & Sort by DEP_DELAY and show top 10 descending
dest_delay = df_flights_nonnull.groupBy('DEST_CITY_NAME').agg(mean('DEP_DELAY')).toPandas()
dest_delay = dest_delay.sort_values(by='avg(DEP_DELAY)', ascending=False).head(10)
sns.barplot(x='DEST_CITY_NAME', y='avg(DEP_DELAY)', data=dest_delay)
plt.title('Top 10 Average Departure Delay by Destination Airport')
plt.xlabel('Destination Airport')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()


# Group & Sort by ARR_DELAY and show top 10 ascending
origin_delay = df_flights_nonnull.groupBy('ORIGIN_CITY_NAME').agg(mean('ARR_DELAY')).toPandas()
origin_delay = origin_delay.sort_values(by='avg(ARR_DELAY)', ascending=True).head(10)
sns.barplot(x='ORIGIN_CITY_NAME', y='avg(ARR_DELAY)', data=origin_delay)
plt.title('Bottom 10 Average Arrival Delay by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

# Group & Sort by DEP_DELAY and show top 10 ascending
dest_delay = df_flights_nonnull.groupBy('DEST_CITY_NAME').agg(mean('DEP_DELAY')).toPandas()
dest_delay = dest_delay.sort_values(by='avg(DEP_DELAY)', ascending=True).head(10)
sns.barplot(x='DEST_CITY_NAME', y='avg(DEP_DELAY)', data=dest_delay)
plt.title('Bottom 10 Average Departure Delay by Destination Airport')
plt.xlabel('Destination Airport')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

## Some airlines have a higher likelihood of getting delayed and some airports are the busiest and some are the very less likely
## Will be also interesting to see what this data shows if we do not consider early arrivals or early departures

# COMMAND ----------

df_flights_nonnullordest = df_flights.select(['OP_UNIQUE_CARRIER','ARR_DELAY', 'DEP_DELAY','ORIGIN_CITY_NAME', 'DEST_CITY_NAME']).na.drop()

route_delay = df_flights_nonnullordest.groupBy('ORIGIN_CITY_NAME', 'DEST_CITY_NAME').agg(mean('ARR_DELAY'), mean('DEP_DELAY')).toPandas()
route_delay['route'] = route_delay['ORIGIN_CITY_NAME'] + ' - ' + route_delay['DEST_CITY_NAME']

### Sort by ARR_DELAY and show top 5
top_5_arr_delay = route_delay.sort_values(by='avg(ARR_DELAY)', ascending=False).head(5)
sns.barplot(x='route', y='avg(ARR_DELAY)', data=top_5_arr_delay)
plt.title('Top 5 Average Arrival Delay by Route')
plt.xlabel('Route')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

#### Sort by DEP_DELAY and show top 5
top_5_dep_delay = route_delay.sort_values(by='avg(DEP_DELAY)', ascending=False).head(5)
sns.barplot(x='route', y='avg(DEP_DELAY)', data=top_5_dep_delay)
plt.title('Top 5 Average Departure Delay by Route')
plt.xlabel('Route')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

# COMMAND ----------

df_flights_nonnullordest = df_flights.select(['OP_UNIQUE_CARRIER','ARR_DELAY', 'DEP_DELAY','ORIGIN_CITY_NAME', 'DEST_CITY_NAME']).na.drop()

route_delay = df_flights_nonnullordest.groupBy('ORIGIN_CITY_NAME', 'DEST_CITY_NAME').agg(mean('ARR_DELAY'), mean('DEP_DELAY')).toPandas()
route_delay['route'] = route_delay['ORIGIN_CITY_NAME'] + ' - ' + route_delay['DEST_CITY_NAME']

### Sort by ARR_DELAY and show top 5
top_5_arr_delay = route_delay.sort_values(by='avg(ARR_DELAY)', ascending=True).head(5)
sns.barplot(x='route', y='avg(ARR_DELAY)', data=top_5_arr_delay)
plt.title('Bottom 5 Average Arrival Delay by Route')
plt.xlabel('Route')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

#### Sort by DEP_DELAY and show top 5
top_5_dep_delay = route_delay.sort_values(by='avg(DEP_DELAY)', ascending=True).head(5)
sns.barplot(x='route', y='avg(DEP_DELAY)', data=top_5_dep_delay)
plt.title('Bottom 5 Average Departure Delay by Route')
plt.xlabel('Route')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

# COMMAND ----------

delay_reasons = df_flights.select(mean('CARRIER_DELAY'), mean('WEATHER_DELAY'), mean('NAS_DELAY'), mean('SECURITY_DELAY'), mean('LATE_AIRCRAFT_DELAY')).toPandas()

### Plot the average delay time for each delay reason
sns.barplot(data=delay_reasons)
plt.title('Average Delay Time by Delay Reason')
plt.xlabel('Delay Reason')
plt.ylabel('Average Delay Time (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()

# COMMAND ----------

origin_counts = df_flights_nonnullordest.groupBy('ORIGIN_CITY_NAME').agg(count('*').alias('count')).toPandas()

top_10_origin_counts = origin_counts.sort_values(by='count', ascending=False).head(10)

## Plot the number of flights by origin airport
sns.barplot(x='count', y='ORIGIN_CITY_NAME', data=top_10_origin_counts)
plt.title('Top 10 Origin Airports by Number of Flights')
plt.xlabel('Number of Flights')
plt.ylabel('Origin Airport')
plt.tight_layout()
plt.show()

### Group the data by destintion airport and count the number of flights
dest_counts = df_flights_nonnullordest.groupBy('DEST_CITY_NAME').agg(count('*').alias('count')).toPandas()

top_10_dest_counts = dest_counts.sort_values(by='count', ascending=False).head(10)

## Plot the number of flights by dest airport
sns.barplot(x='count', y='DEST_CITY_NAME', data=top_10_dest_counts)
plt.title('Top 10 Destination Airports by Number of Flights')
plt.xlabel('Number of Flights')
plt.ylabel('Destination Airport')
plt.tight_layout()
plt.show()

# COMMAND ----------

display(df_flights_a[df_flights_a.CRS_ARR_TIME.cast('integer') < 200])

# COMMAND ----------

from pyspark.sql.functions import substring, lpad
## 24 Hour format based delay analysis

df_flights_a = df_flights.withColumn('CRS_ARR_TIME', lpad(df_flights.CRS_ARR_TIME, 4, '0'))
df_flights_d = df_flights.withColumn('CRS_DEP_TIME', lpad(df_flights.CRS_DEP_TIME, 4, '0'))

df_flights_a = df_flights_a.select(['ARR_DELAY','CRS_ARR_TIME']).withColumn('arr_hour', substring(df_flights_a.CRS_ARR_TIME, 0, 2))

df_flights_d = df_flights_d.select(['DEP_DELAY','CRS_DEP_TIME']).withColumn('dep_hour', substring(df_flights_d.CRS_DEP_TIME, 0, 2))

avg_arr_delay_by_hour = df_flights_a.groupBy('arr_hour').agg(mean('ARR_DELAY').alias('avg_arr_delay')).toPandas()

# Convert the arr_hour column to an integer and sort the data by hour
avg_arr_delay_by_hour['arr_hour'] = avg_arr_delay_by_hour['arr_hour'].astype(int)
avg_arr_delay_by_hour = avg_arr_delay_by_hour.sort_values(by='arr_hour')

### Plot the average arrival delay by hour
ax = sns.barplot(x='arr_hour', y='avg_arr_delay', data=avg_arr_delay_by_hour)
plt.title('Average Arrival Delay by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Arrival Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()


avg_dep_delay_by_hour = df_flights_d.groupBy('dep_hour').agg(mean('DEP_DELAY').alias('avg_dep_delay')).toPandas()
avg_dep_delay_by_hour['dep_hour'] = avg_dep_delay_by_hour['dep_hour'].astype(int)
avg_dep_delay_by_hour = avg_dep_delay_by_hour.sort_values(by='dep_hour')

### Plot the average departure delay by hour
ax = sns.barplot(x='dep_hour', y='avg_dep_delay', data=avg_dep_delay_by_hour)
plt.title('Average Departure Delay by Hour')
plt.xlabel('Hour')
plt.ylabel('Average Departure Delay (minutes)')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()
plt.show()


## NOte: there is something wrong with the data here and I will invetigate.

# COMMAND ----------

# Calculate # of nulls in each column
from pyspark.sql.functions import col,isnan,when,count
display(df_flights.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_flights.columns]
   ).toPandas())

# COMMAND ----------

#Calculate how many nulls are in each column  
from pyspark.sql.functions import col,isnan,when,count
df_flights.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df_flights.columns]
   ).toPandas()

# COMMAND ----------

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

import pyspark.sql.functions as F
import matplotlib.pyplot as plt

df_weather = spark.read.parquet(f"{data_BASE_DIR}parquet_weather_data_3m/")

# Convert columns to appropriate data types
df_weather = df_weather.withColumn("HourlyPrecipitationDouble", F.col("HourlyPrecipitation").cast("double"))
df_weather = df_weather.withColumn("HourlyVisibilityDouble", F.col("HourlyVisibility").cast("double"))
df_weather = df_weather.withColumn("HourlyWindSpeedDouble", F.col("HourlyWindSpeed").cast("double")).filter(col("HourlyWindSpeedDouble") < 2000)

# Overlayed boxplots for df_weather
weather_cols = ['HourlyPrecipitationDouble', 'HourlyVisibilityDouble', 'HourlyWindSpeedDouble']
weather_data = df_weather.select(*weather_cols).toPandas()

plt.figure(figsize=(10, 6))
weather_data.boxplot(column=weather_cols)
plt.title('Boxplots of Weather Variables')
plt.xlabel('Weather Variables')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()

# Grouping and aggregation for df_stations
grouped_stations = df_stations.groupBy('neighbor_id').agg(
    F.avg('distance_to_neighbor').alias('avg_distance_to_neighbor'),
).orderBy('avg_distance_to_neighbor')

display(grouped_stations)

# Grouping and aggregation for df_flights
grouped_flights = df_flights.groupBy('OP_UNIQUE_CARRIER').agg(
    F.avg('DEP_DELAY').alias('Avg_DEP_DELAY'),
    F.avg('ARR_DELAY').alias('Avg_ARR_DELAY'),
    F.avg('DISTANCE').alias('Avg_DISTANCE')
)

display(grouped_flights)

# COMMAND ----------

display(df_weather)

# COMMAND ----------

print(df_weather.columns)

# COMMAND ----------

from pyspark.sql.functions import to_date

df_weather_t = df_weather.withColumn('DATE', to_date(df_weather.DATE))

avg_temp_by_month = df_weather_t.groupBy(month('DATE')).agg(mean('DailyAverageDryBulbTemperature').alias('avg_temp')).toPandas()
### Plot the average temperature by month
sns.barplot(x='month(DATE)', y='avg_temp', data=avg_temp_by_month)
plt.title('Avg Temperature by Month')
plt.xlabel('Month')
plt.ylabel('Avg Temperature (F)')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC # ML Algs and Metrics
# MAGIC
# MAGIC In order to measure success of predicting flight delays, with the assumption that we are predicting a binary outcome of either on-time or delay, we want to make sure we are measuring more than just accuracy, as incorrectly predicting delays would negatively impact those who are flying. Additionally, there will be a class imbalance between delayed and non-delayed flights since most flights are not delayed so it makes accuracy not a strong metric. Thus, our main metric will be AUC, but we will also use precision and recall as well. A high AUC will be consistent with accurately predicting delays with few false positives, which is our ideal outcome.
# MAGIC
# MAGIC For machine learning algorithms, we propose to focus on both a more traditional ensemble approach as well as a deep learning approach. Our ensemble method will be a mixture of Random Forests and Gradient-Boosted Trees, as we can see whether or not boosting or bootstrap aggregation is a more effective approach. Since there will be a lot of available features, we believe a decision tree-based method may have some success.
# MAGIC
# MAGIC For the deep learning approach, we are going to use LSTMs, as they are known to be fairly adept when it comes to handling time-series data. There are a few CPU implementations for LSTMs on databricks, which we will be using since we do not have GPU access.
# MAGIC
# MAGIC When implementing the ML and the Deep Learning algorithms we will be implementing cross-validation as well, to make sure that performance is not dependent on the selected training set. For Deep Learning, we will use binary cross entropy as a loss function as it closely mirrors AUC, the metric we are hoping to maximize.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Pipeline
# MAGIC
# MAGIC ![Block Diagram](files/tables/261_Block_Diagram-3.png)
