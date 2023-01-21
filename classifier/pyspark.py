# Execution for immune.txt
import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import preproc as pp

#import preproc as pp
'''
check_lang_udf = udf(check_lang, StringType())
'''
remove_stops_udf = udf(pp.remove_stops, StringType())
remove_features_udf = udf(pp.remove_features, StringType())
tag_and_remove_udf = udf(pp.tag_and_remove, StringType())
lemmatize_udf = udf(pp.lemmatize, StringType())
check_blanks_udf = udf(pp.check_blanks, StringType())

# create Spark context with Spark configuration

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)
'''
# import
from pyspark.sql import HiveContext
sc = pyspark.SparkContext()
sqlContext = HiveContext(sc)
#spark = SparkSession.builder.appName("DataFrame").getOrCreate()
'''
# Uncomment here
# Read file into RDD

test_rdd = sc.textFile("2010-neg.txt")

#parts_rdd = data_rdd.map(lambda l: l.split(","))
# Each line is converted to a tuple.
#tuple_rdd = parts_rdd.map(lambda p: Row(id=int(p[0]), order_date=p[1], customer_id=p[2], order_status=p[3]))
#filled_rdd = tuple_rdd.filter(bool)


#lines.toDF()
#lines = spark.read.text("immune.txt")
# Call collect() to get all data
#llist = lines.collect()

# print line one by line
#for line in llist:
#	print(line)
# import
'''
# Then, you can use the com.databricks.spark.csv.
lines = sqlContext.read \
     .format('com.databricks.spark.csv') \
     .options(header='false', delimiter='|') \
     .load('immune.txt')
'''
parts_rdd = test_rdd.map(lambda l: l.split("\t"))
# Filter bad rows out
filled_rdd = parts_rdd.filter(bool)
#typed_rdd = data_rdd.map(lambda p: (p[0], p[1], float(p[2])))

#Create DataFrame
data_df = sqlContext.createDataFrame(test_rdd, StringType())
'''
data_df = sqlContext.createDataFrame(prob_rdd, StringType())
data_df = sqlContext.createDataFrame(imm_rdd, StringType())
data_df = sqlContext.createDataFrame(trans_rdd, StringType())
data_df = sqlContext.createDataFrame(risk_rdd, StringType())
'''
# Ignore
#label_df = data_df.withColumn('label', )
#full_df = data_df.withColumn('id',monotonically_increasing_id())


# get the raw columns
raw_cols = data_df.columns
#raw_cols = lines.columns


data_df.printSchema()
#lines.printSchema()
data_df.na.drop(how="any").show(truncate=False)

# Remove stopwords from values
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords.words('english')
rm_stops_df = data_df.select(raw_cols).withColumn("stop_text", remove_stops_udf(data_df["value"]))
rm_stops_df.show(20)

# Remove features from values
import re
import string
rm_features_df = rm_stops_df.select(raw_cols+["stop_text"]).withColumn("feat_text", remove_features_udf(rm_stops_df["stop_text"]))
rm_features_df.show(20)

# Tagging text from value
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
tagged_df = rm_features_df.select(raw_cols+["feat_text"]).withColumn("tagged_text", tag_and_remove_udf(rm_features_df.feat_text))
tagged_df.show(20)

# Lemmatization for values
import re
import string
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
lemm_df = tagged_df.select(raw_cols+["tagged_text"]).withColumn("lemm_text", lemmatize_udf(tagged_df["tagged_text"]))
lemm_df.show(20)

check_blanks_df = lemm_df.select(raw_cols+["lemm_text"]).withColumn("is_blank", check_blanks_udf(lemm_df["lemm_text"]))
# remove blanks from lemmatized text
no_blanks_df = check_blanks_df.filter(check_blanks_df["is_blank"] == "False")

# drop duplicates from lemmatized text
dedup_df = no_blanks_df.dropDuplicates(['lemm_text'])

dedup_df.show(20)

# Rename final column as text for processing
final_df = dedup_df.selectExpr("lemm_text as text")
final_df.show()
final_df.printSchema()

from pyspark.sql.functions import monotonically_increasing_id
# Create Unique ID for text
uid_df = final_df.withColumn("uid", monotonically_increasing_id())
uid_df.show(4)

# Create label with 1.0 being the constant for ML classifier
from pyspark.sql.functions import udf

@udf("double")
def const_col():
    return 1.0

label_df = uid_df.withColumn('label', const_col())
label_df.show()

# Split the data into training and test sets (40% held out for testing)
(trainingData, testData) = data.randomSplit([0.6, 0.4])

# Running Naive Bayes classifier.
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
#data = tokenizer.transform(data)
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
#labelEncoder = StringIndexer(inputCol="text", outputCol='label')
#vectorizer = CountVectorizer(inputCol= "words", outputCol="rawFeatures")
idf = IDF(minDocFreq=2000, inputCol="rawFeatures", outputCol="features")
#assembler = VectorAssembler(inputCols=["hour", "mobile", "userFeatures"],outputCol="features")
#idfModel = idf.fit(data)

#lda = LDA(k=20, seed=1, optimizer="em")

''' DecisionTree Classifier
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
'''
'''
# Linear Support Vector Classifer
lsvc = LinearSVC((maxIter=10, regParam=0.1)
'''
# Naive Bayes model
nb = NaiveBayes(smoothing=2.0)
'''
# Random Forest Classifier
rfc = RandomForestClassifier()
'''
# Pipeline Architecture for NB
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])
'''
# Pipeline Architecture for RFC
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, rfc])
pipeline

# Pipeline Architecture for DT
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, dt])
pipeline

# Pipeline Architecture for LSVC
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lsvc])
pipeline
'''


# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)
model

predictions = model.transform(testData)

# Select example rows to display.
predictions.select("text", "label", "prediction").show(5,True)

from pyspark.ml.evaluation import RegressionEvaluator
r_evaluator = RegressionEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
m_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
b_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
evaluator.evaluate(predictions)

from pyspark.sql.types import FloatType

from textblob import TextBlob

def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

sentiment_analysis_udf = udf(sentiment_analysis , FloatType())

df  = label_df.withColumn("sentiment_score", sentiment_analysis_udf( label_df['text'] ))
df.show(20,True)
'''
# Pipeline for LDA model
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.clustering import LDA
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
#data = tokenizer.transform(data)
vectorizer = CountVectorizer(inputCol= "words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")
#idfModel = idf.fit(data)

lda = LDA(k=20, seed=1, optimizer="em")
lda
pipeline = Pipeline(stages=[tokenizer, vectorizer,idf, lda])


model = pipeline.fit(data)
'''