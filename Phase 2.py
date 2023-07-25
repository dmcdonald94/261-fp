# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Credit Assignment Table
# MAGIC
# MAGIC # Phase Leader: Rithvik
# MAGIC
# MAGIC # Video Link
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # EDA
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Loading the data

# COMMAND ----------

# imports
import seaborn as sns

# COMMAND ----------

# define base directory that contains all the data
data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# data set of interest
display(dbutils.fs.ls(f"{data_BASE_DIR}/OTPW_3M"))

# COMMAND ----------

# read in the data from compressed format
raw_df = spark.read.format('csv').option("compression", "gzip").option("header", "true").option("inferSchema", "true").load(f"{data_BASE_DIR}OTPW_3M//OTPW_3M_2015.csv.gz")
raw_df.count()

# COMMAND ----------

raw_df.printSchema()

# COMMAND ----------

print(f"# of columns = {len(raw_df.columns)}")

# COMMAND ----------

input_features_df = raw_df.select(*raw_df.columns)

# Step 2: Find the families of input features by extracting the common prefixes
input_feature_cols = input_features_df.columns
families = set()
for col in input_feature_cols:
    for i in range(1, len(col)):
        families.add(col[:i])

# Step 3: Count the occurrence of each family
family_count = {}
for col in input_feature_cols:
    for family in families:
        if col.startswith(family):
            family_count[family] = family_count.get(family, 0) + 1

# Step 4: Print the families and their counts
for family, count in family_count.items():
    print(f"Family: {family}, Count: {count}")

# COMMAND ----------

# look at some statistics for label of interest
LABEL = "ARR_DELAY"
from pyspark.sql.functions import mean, stddev, min, max, isnan, count, when, col
display(raw_df.select(LABEL).select(mean(LABEL), stddev(LABEL), min(LABEL), max(LABEL)))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Above results show that the average delay is around 6.2 minutes, with a standard deviation of 40.5 minutes. The most a flight has been delayed is 998.0 minutes (yikes). Is this an actual value or is it a filler with some other meaning?
# MAGIC
# MAGIC Since we are only interested in predicting whether or not a flight will be delayed by 15 minutes or more, there is a convenient column labled `ARR_DEL15` where `1` indicates that a delay was in fact greater than 15 minutes.
# MAGIC
# MAGIC We will use the next code block to ensure the meaning of this label.

# COMMAND ----------

LABEL = "ARR_DEL15"
checked_df = raw_df.withColumn("MY_ARR_DEL15", when(raw_df['ARR_DELAY'] >= 15.0, 1).otherwise(0))

checked_df = checked_df.withColumn("arr_del15_correct", col("MY_ARR_DEL15") == col("ARR_DEL15")).drop("MY_ARR_DEL15").drop("ARR_DEL15")
if (checked_df.filter(checked_df["arr_del15_correct"] == False).count() == 0):
    print("ARR_DEL15 is correct.")
else:
    print("There is a discrepancy in ARR_DEL15.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Missing Data
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Labels

# COMMAND ----------

# look at rows with missing label
print("# of missing values in label column.")
raw_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in [LABEL]]).show()



# COMMAND ----------

# since we are missing labels for these rows, we should just drop the entire row.
initial_count = raw_df.count()
df = raw_df.dropna(subset=LABEL)
final_count = df.count()
print(f"{(1 - final_count / initial_count) * 100}% of data was missing label, and dropped.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Features

# COMMAND ----------

missing_features_df = df.select(*[
    (
        count(when((isnan(c) | col(c).isNull()), c)) if t not in ("timestamp")
        else count(when(col(c).isNull(), c))
    ).alias(c)
    for c, t in df.dtypes if c in df.columns
]).cache()

display(missing_features_df)

# COMMAND ----------


missing_features_df = missing_features_df.withColumns({name: missing_features_df[name] / final_count * 100 for name in missing_features_df.columns}).cache()


display(missing_features_df)

# COMMAND ----------



# COMMAND ----------

cols_missing = [col_name for col_name in missing_features_df.columns if not missing_features_df.filter(col(col_name) == 0).count()]

# for col in missing_features_df.columns:
#     if missing_features_df.select(col) != 0:
#         cols_with_missing.append(col)


# COMMAND ----------

missing_features_df = missing_features_df.select(*cols_missing)
display(missing_features_df)


# COMMAND ----------

from pyspark.sql.functions import concat_ws, to_timestamp

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY") 

# COMMAND ----------

df.columns

# COMMAND ----------

df = df.withColumn("datetime_str", concat_ws(' ', col("FL_DATE"), col("CRS_DEP_TIME")))
df = df.withColumn("flight_datetime", to_timestamp(col("datetime_str"),"yyyy-MM-dd Hmm"))
df = df.withColumn("weather_datetime", to_timestamp(col("DATE"))) 
df.select(["flight_datetime", "FL_DATE", "CRS_DEP_TIME", "DEP_DELAY"]).take(4)

# COMMAND ----------

# FEATURES TO USE
FEATURES_COLS = ['flight_datetime', 'weather_datetime', 'HourlyPrecipitation', 'HourlyVisibility', 'HourlyWindGustSpeed', 'HourlyWindSpeed']
LABEL = ['DEP_DEL15']

COLS_TO_USE = FEATURES_COLS + LABEL
# now lets only use these columns
experiment_df = df.select(COLS_TO_USE)

# COMMAND ----------

experiment_df.select("*").take(1)

# COMMAND ----------

from pyspark.sql.types import FloatType

experiment_df = experiment_df.withColumns({name: col(name).cast('float') for name in experiment_df.columns[2:]}).cache()

# COMMAND ----------

experiment_df.select("*").take(2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Baseline Pipeline on all data
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Cross Validation
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import window, sum

# COMMAND ----------

experiment_df.printSchema()

# COMMAND ----------

FEATURES_COLS[2:]

# COMMAND ----------

# define windows
# df.select(["flight_datetime","weather_datetime"])

# w = experiment_df.groupby(window("weather_datetime", "4 hours")).agg(mean("HourlyPrecipitation").alias("mean"))
# w.select(w.window.start.cast("string").alias("start"),
#                w.window.end.cast("string").alias("end"), "mean").collect()



from pyspark.sql.window import Window
days = lambda i: i * 86400
hours = lambda i: i * 3600


w = (Window.orderBy(col("weather_datetime").cast('long')).rangeBetween(-hours(6), -hours(2)))
# w = (Window.orderBy(col("weather_datetime").cast("long")).rangeBetween(-hours(6), -hours(2)))

print(f"{len(experiment_df.columns) = }")
print(f"{experiment_df.count() = }")

for column in FEATURES_COLS[2:]:
    experiment_df = experiment_df.withColumn(f"rolling_{column}_average", mean(column).over(w))    
print(f"{len(experiment_df.columns) = }")
print(f"{experiment_df.count() = }")

FEATURES_COLS = [f"rolling_{column}_average"  if column.find("_datetime") == -1 else column for column in FEATURES_COLS]





# COMMAND ----------

missing_exp_df = experiment_df.select(*[
    (
        count(when((isnan(c) | col(c).isNull()), c)) if t not in ("timestamp", "date")
        else count(when(col(c).isNull(), c))
    ).alias(c)
    for c, t in experiment_df.dtypes if c in FEATURES_COLS
])

display(missing_exp_df)


# COMMAND ----------

# lets just drop nans to get things to run.
exp_df_count = experiment_df.count()
print(f"{exp_df_count = }")
dropped_exp_df = experiment_df.na.drop()
dropped_exp_df_count = dropped_exp_df.count()
print(f"{dropped_exp_df_count = }")
print(f"Dropped {dropped_exp_df_count / exp_df_count * 100}% of data.")

# COMMAND ----------

from pyspark.ml.feature import StandardScaler, VectorAssembler
vectorAssembler = VectorAssembler(inputCols = FEATURES_COLS[2:], outputCol = 'features', handleInvalid='skip')
texperiment_df = vectorAssembler.transform(experiment_df)
texperiment_df.select(["features"]).show(3)





# COMMAND ----------

standardScaler = StandardScaler()
standardScaler.setInputCol("features")
standardScaler.setOutputCol("features_scaled")
model = standardScaler.fit(texperiment_df)
texperiment_df = model.transform(texperiment_df)


# COMMAND ----------



# COMMAND ----------

LABEL

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
lr = LogisticRegression(featuresCol='features_scaled', labelCol = LABEL[0])

lr.fit(texperiment_df)
# grid = ParamGridBuilder().addGrid(lr.maxIter, [0,1,5]).build()
# evaluator = BinaryClassificationEvaluator()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Scalability

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Efficiency
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
