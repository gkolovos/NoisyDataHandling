# -*- coding: utf-8 -*-

## its the second part of the script
## the first part creates noise and null and outliers in a given dataset
## in this part we take as input the first noisy dataset that we created (out.csv)
## the out.csv can be found on https://github.com/gkolovos/kaggleDATAset
## in this part we handle  the outliers and the null valies
## then we upply the Grid Search CV model we created in order the evaluate its metrics


#install pyspark
! pip install pyspark

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import seaborn as sns

spark = SparkSession.builder.appName('Heart_Disease_pySpark').getOrCreate()
spark

! git clone https://github.com/gkolovos/kaggleDATAset

ls kaggleDATAset

#create spark dataframe
df = spark.read.csv("/content/kaggleDATAset/out.csv", header=True, inferSchema=True )

df.show()

### spark df to pandas df 

pandasDF = df.toPandas()

print(pandasDF)

x_noisy = pandasDF

######## final dataframe with noise, outliers and NaN times ###########################

print(x_noisy)

x_noisy.dtypes

print(x_noisy)

##### we can fill the NaN values using the mean or median values
### for better results we can use the KNN algorithm

### the next step is to remove the outliners and the null values in order to run the algorithm model and see the results

################################################################ detect outliers ##########################################################################

import numpy as np
import pandas as pd

x_noisy.sample(5)

x_noisy.describe()

# printing the whole dataset

# pd.set_option('display.max_rows', x_noisy.shape[0]+1)
# print(x_noisy)

##  Plot the distribution plot of “Age” feature

sns.distplot(x_noisy['Age'])

# Plot the box-plot of “Age” feature

sns.boxplot(x_noisy['Age'])

print(x_noisy['Age'].quantile(0.50))   ### median timh

#  Finding upper and lower limit

upper_limit = x_noisy['Age'].quantile(0.99)
lower_limit = x_noisy['Age'].quantile(0.01)

lower_limit

upper_limit

## first choise
# Apply trimming
## x_noisy1 = x_noisy[(x_noisy['Age'] <= 73.0) & (x_noisy['Age'] >= 32.0)]
# this technique deletes all rows tha consist of outliers

# we will replace outliers using the upper and lower limit 

## second choise
 
## filing outliers with the median timh
# x_noisytest['Age'] = x_noisytest.where(x_noisytest['Age'] > upper_limit, 54.0, x_noisytest['Age'])
# x_noisy.describe()

## third choise
 
## filing outliers with the upper_limit and lower limit
#x_noisytest['Age'] = x_noisytest.where(x_noisytest['Age'] > upper_limit , 73, x_noisytest['Age'])
#x_noisy.describe()

x_noisy['Age'] = np.where(x_noisy['Age'] >= upper_limit,
        74.0 ,
        np.where(x_noisy['Age'] <= lower_limit,
        31.0 ,
        x_noisy['Age']))

# Compare the distribution and box-plot after trimming

sns.distplot(x_noisy['Age'])
sns.boxplot(x_noisy['Age'])

sns.distplot(x_noisy['Age'])

x_noisy.groupby(['Age']).size()

x_noisy.isna().sum()

count_nan = len(x_noisy) - x_noisy.count()
count_nan

print(x_noisy)

# print all rows

# pd.set_option('display.max_rows', x_noisy.shape[0]+1)
# print(x_noisy)

# sex

# values vary between 0 and 1 so its ok

# ChestPainType

sns.distplot(x_noisy['ChestPainType'])

sns.boxplot(x_noisy['ChestPainType'])

upper_limit = x_noisy['ChestPainType'].quantile(0.99)
lower_limit = x_noisy['ChestPainType'].quantile(0.05)

upper_limit

lower_limit

x_noisy['ChestPainType'] = np.where(x_noisy['ChestPainType'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['ChestPainType'] <= lower_limit,
        lower_limit,
        x_noisy['ChestPainType']))

#sns.distplot(x_noisyNum2['ChestPainType'])
sns.boxplot(x_noisy['ChestPainType'])

sns.distplot(x_noisy['ChestPainType'])

x_noisy.groupby(['ChestPainType']).size()

x_noisy.describe()

# RestingBP

sns.distplot(x_noisy['RestingBP'])

sns.boxplot(x_noisy['RestingBP'])

upper_limit = x_noisy['RestingBP'].quantile(0.99)
lower_limit = x_noisy['RestingBP'].quantile(0.01)

lower_limit

upper_limit

x_noisy['RestingBP'] = np.where(x_noisy['RestingBP'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['RestingBP'] <= lower_limit,
        lower_limit,
        x_noisy['RestingBP']))

sns.distplot(x_noisy['RestingBP'])
sns.boxplot(x_noisy['RestingBP'])

##### we could redo this step for higher accuracy but its enough for now

sns.distplot(x_noisy['RestingBP'])

x_noisy.describe()

# Cholesterol

sns.distplot(x_noisy['Cholesterol'])

sns.boxplot(x_noisy['Cholesterol'])

upper_limit = x_noisy['Cholesterol'].quantile(0.99)
lower_limit = x_noisy['Cholesterol'].quantile(0.10)

upper_limit

lower_limit

x_noisy['Cholesterol'] = np.where(x_noisy['Cholesterol'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['Cholesterol'] <= lower_limit,
        lower_limit,
        x_noisy['Cholesterol']))

sns.distplot(x_noisy['Cholesterol'])
sns.boxplot(x_noisy['Cholesterol'])

sns.distplot(x_noisy['Cholesterol'])

x_noisy.groupby(['Cholesterol']).size()

x_noisy.describe()

## FastingBS

sns.distplot(x_noisy['FastingBS'])

sns.boxplot(x_noisy['FastingBS'])

upper_limit = x_noisy['FastingBS'].quantile(0.90)
lower_limit = x_noisy['FastingBS'].quantile(0.10)

lower_limit

upper_limit

x_noisy['FastingBS'] = np.where(x_noisy['FastingBS'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['FastingBS'] <= lower_limit,
        lower_limit,
        x_noisy['FastingBS']))

#sns.distplot(x_noisy['FastingBS'])
sns.boxplot(x_noisy['FastingBS'])

sns.distplot(x_noisy['FastingBS'])

x_noisy.describe()

## RestingECG

sns.distplot(x_noisy['RestingECG'])

sns.boxplot(x_noisy['RestingECG'])

upper_limit = x_noisy['RestingECG'].quantile(0.99)
lower_limit = x_noisy['RestingECG'].quantile(0.01)

upper_limit

lower_limit

x_noisy['RestingECG'] = np.where(x_noisy['RestingECG'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['RestingECG'] <= lower_limit,
        lower_limit,
        x_noisy['RestingECG']))

#sns.distplot(x_noisy['RestingECG'])
sns.boxplot(x_noisy['RestingECG'])

sns.distplot(x_noisy['RestingECG'])

x_noisy.groupby(['RestingECG']).size()

x_noisy.describe()

### MaxHR

sns.distplot(x_noisy['MaxHR'])

sns.boxplot(x_noisy['MaxHR'])

upper_limit = x_noisy['MaxHR'].quantile(0.99)
lower_limit = x_noisy['MaxHR'].quantile(0.01)

lower_limit

upper_limit

x_noisy['MaxHR'] = np.where(x_noisy['MaxHR'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['MaxHR'] <= lower_limit,
        lower_limit,
        x_noisy['MaxHR']))

sns.distplot(x_noisy['MaxHR'])
sns.boxplot(x_noisy['MaxHR'])

sns.distplot(x_noisy['MaxHR'])

x_noisy.describe()

## ExerciseAngina

sns.distplot(x_noisy['ExerciseAngina'])

sns.boxplot(x_noisy['ExerciseAngina'])

upper_limit = x_noisy['ExerciseAngina'].quantile(0.95)
lower_limit = x_noisy['ExerciseAngina'].quantile(0.05)

lower_limit

upper_limit

x_noisy['ExerciseAngina'] = np.where(x_noisy['ExerciseAngina'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['ExerciseAngina'] <= lower_limit,
        lower_limit,
        x_noisy['ExerciseAngina']))

sns.distplot(x_noisy['ExerciseAngina'])
#sns.boxplot(x_noisy['ExerciseAngina'])

x_noisy.describe()

## Oldpeak

sns.distplot(x_noisy['Oldpeak'])

sns.boxplot(x_noisy['Oldpeak'])

upper_limit = x_noisy['Oldpeak'].quantile(0.99)
lower_limit = x_noisy['Oldpeak'].quantile(0.01)

lower_limit

upper_limit

x_noisy['Oldpeak'] = np.where(x_noisy['Oldpeak'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['Oldpeak'] <= lower_limit,
        lower_limit,
        x_noisy['Oldpeak']))

sns.distplot(x_noisy['Oldpeak'])
sns.boxplot(x_noisy['Oldpeak'])

sns.distplot(x_noisy['Oldpeak'])

x_noisy.describe()

# ST_Slope

sns.distplot(x_noisy['ST_Slope'])

sns.boxplot(x_noisy['ST_Slope'])

upper_limit = x_noisy['ST_Slope'].quantile(0.99)
lower_limit = x_noisy['ST_Slope'].quantile(0.01)

lower_limit

upper_limit

x_noisy['ST_Slope'] = np.where(x_noisy['ST_Slope'] >= upper_limit,
        upper_limit,
        np.where(x_noisy['ST_Slope'] <= lower_limit,
        lower_limit,
        x_noisy['ST_Slope']))

#sns.distplot(x_noisy['ST_Slope'])
sns.boxplot(x_noisy['ST_Slope'])

sns.distplot(x_noisy['ST_Slope'])

x_noisy.groupby(['ST_Slope']).size()

x_noisy.describe()

print(x_noisy)

#### outliers have been removed

## now we should replace the NaN values

##################################3####  Replace NaN times ##########################################

#### fill NaN values

print(x_noisy)

x_noisy.describe()

## creating testdf to test on how to fill NaN values
# testdf1 = x_noisy

# print(testdf1)

# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# testdf = pd.DataFrame(imputer.fit_transform(testdf),columns = testdf.columns)

x_noisy.isna().any()

# There are different ways to handle missing data. Some methods such as removing the entire observation if it has a missing value or replacing 
# the missing values with mean, median or mode values. However, these methods can waste valuable data or reduce the variability of your dataset. 
# In contrast, KNN Imputer maintains the value and variability of your datasets and yet it is more precise and efficient than using the average 
# values.


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
    
    
imputer = KNNImputer(n_neighbors=6)
imputer.fit_transform(x_noisy)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    

categorical = [ 'sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina' , 'ST_Slope']                  
numerical = ['Age', 'Cholesterol', 'RestingBP',  'MaxHR', 'Oldpeak']                  

x_noisy[categorical] = x_noisy[categorical].apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index
))

# numeric using mean and the rest using most frequent
imp_num = IterativeImputer(estimator=RandomForestRegressor(),initial_strategy='mean',max_iter=10, random_state=0)
imp_cat = IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent',max_iter=10, random_state=0)
    
x_noisy[numerical] = imp_num.fit_transform(x_noisy[numerical])
x_noisy[categorical] = imp_cat.fit_transform(x_noisy[categorical])

print(x_noisy)

x_noisy.describe()

print(x_noisy)

x_noisy.groupby(['ChestPainType']).size()

x_noisy.to_csv("/content/kaggleDATAset/noisyfile2.csv",index=False)

dfn = spark.read.csv("/content/kaggleDATAset/noisyfile2.csv", header=True, inferSchema=True )

dfn.show()

#### Finally i going to use a ML model i created in order to check the denoising model

train_data2,validation_test2 = dfn.randomSplit([0.8,0.2],seed=100)
validation2, test2 = validation_test2.randomSplit([0.5, 0.5], seed = 4)

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Bucketizer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, NGram, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes,DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LinearSVC

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=[
 'sex',
 'Age',
 'RestingBP',
 'Oldpeak',
 'ExerciseAngina',
 'RestingECG',
 'FastingBS',
 'ST_Slope',
 'MaxHR',
 'Cholesterol',
 'ChestPainType'],outputCol='features')

# Import libraries
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline, PipelineModel

pipeline = Pipeline(stages=[])

basePipeline =[assembler]

#lr = LogisticRegression(labelCol="HeartDisease", featuresCol="features", maxIter=20)
lr = LogisticRegression(maxIter=10,labelCol='HeartDisease',featuresCol='features')
pipeline_LR = basePipeline + [lr]
pg_lr = (ParamGridBuilder()
          .baseOn({pipeline.stages: pipeline_LR})
          .addGrid(lr.regParam,[0.01, .08])
          .addGrid(lr.elasticNetParam,[0.1, 0.8])
          .build())

rf = RandomForestClassifier(labelCol='HeartDisease',featuresCol='features',numTrees=50)
pl_rf = basePipeline + [rf]
pg_rf = (ParamGridBuilder()
      .baseOn({pipeline.stages: pl_rf})
      .addGrid(rf.numTrees, [3, 10])
      .build())

lsvcCV = LinearSVC(maxIter=10,regParam=0.1,featuresCol="features", labelCol='HeartDisease')
pl_lsvc = basePipeline + [lsvcCV]
pg_lsvc = (ParamGridBuilder()
       .baseOn({pipeline.stages: pl_rf})
       .addGrid(lsvcCV.regParam, [0.01,0.1,10.0,100.0])
       .addGrid(lsvcCV.maxIter, [10, 100, 1000])
       .build())

paramGrid =pg_rf+pg_lr+pg_lsvc

cvMM = (CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(MulticlassClassificationEvaluator(labelCol="HeartDisease", predictionCol="prediction", metricName="accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10))

cvModel = cvMM.fit(train_data2)

predictionsMM = cvModel.transform(validation_test2)

predictionsMM.groupBy("prediction").count().show()

from sklearn.metrics import roc_curve , auc
from pyspark.ml.functions import vector_to_array


predictionsMM=cvModel.transform(validation_test2)
y_score = predictionsMM.select(vector_to_array("rawPrediction")[1]).rdd.keys().collect()   
y_true = predictionsMM.select("HeartDisease").rdd.keys().collect()
fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = auc(fpr, tpr)

print("Area under ROC Curve: {:.4f}".format(auc))

from sklearn import metrics
import matplotlib.pyplot as plt

plt.figure()
plt.plot([0,1], [0,1], 'k--', color='orange')
plt.plot(fpr, tpr, label='auc = {:.3f}'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

#accuracy

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="HeartDisease", predictionCol="prediction", metricName="accuracy")

MM_acc = acc_evaluator.evaluate(predictionsMM)
print("MM accuracy: {:.4f}".format(MM_acc*100))

# use MulticlassClassificationEvaluator to get f1 scores
evaluator1 = MulticlassClassificationEvaluator(labelCol="HeartDisease")

# use BinaryClassificationEvaluator to get area under PR curve
evaluator2 = BinaryClassificationEvaluator( rawPredictionCol="prediction", labelCol="HeartDisease")

# make evaluation and print f1 and area under PR score per model
f1 = evaluator1.evaluate(predictionsMM, {evaluator1.metricName:'f1'})
print("F1 score on validation: {:.4f} ".format(f1))

pr = evaluator2.evaluate(predictionsMM, {evaluator2.metricName:'areaUnderPR'})
print("Area under PR on validation set: {:.4f} ".format(pr))

### Below there are the results of the same ML model on the dataset without noise

# auc :       0.94
# accuracy :  88.9
# f1 :        0.89
# PR :        89.3
### As it should be expected the metrics are slightly higher
### However the fact that there is such a small deviation means that our denoise model is a great model

