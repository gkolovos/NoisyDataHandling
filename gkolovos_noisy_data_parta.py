# -*- coding: utf-8 -*-

## this script creates a new noisy dataset with null values and outliers included
## the new dataset is created and was given the name out.csv
## in order to fast forward the process, a copy of the first file that we created was uploaded to https://github.com/gkolovos/kaggleDATAset
## the out.csv file in that path is the file that we used in our work in order to perform the denoising model
## please note that this script creates random noise in the whole dataset so keep in mind that if re-run the script a new out.csv file will be created
## for our convenience we uploaded the file we used in our work on github


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
df = spark.read.csv("/content/kaggleDATAset/heart.csv", header=True, inferSchema=True )

df.show()

##### converting categorical values (F or M, TA or ASY etc etc) into numeric ones

#immportance of chest pain type
def ChestPainType_importance(col):
    if col == 'NAP':
        return 0.0
    elif col == 'ASY':
        return 0.5
    elif col == 'ATA':
        return 0.75
    else:
         return 1.0

ChestPainType_udf = udf(lambda val: ChestPainType_importance(val), DoubleType())
df_after_edit_ChestPainType = df.withColumn('ChestPainType',ChestPainType_udf('ChestPainType'))

df_after_edit_ChestPainType.show(10)

def ExerciseAngina_val(col):
    if col == 'Y':
        return 1
    else:
        return 0

ExerciseAngina_udf = udf(lambda val: ExerciseAngina_val(val), IntegerType())
df_after_edit_ExerciseAngina = df_after_edit_ChestPainType.withColumn('ExerciseAngina', ExerciseAngina_udf('ExerciseAngina'))

def sex_val(col):
    if col == 'F':
        return 0
    else:
        return 1

sex_udf = udf(lambda val: sex_val(val), IntegerType())
df_after_edit_sex = df_after_edit_ExerciseAngina.withColumn('sex', sex_udf('sex'))

final_df = df_after_edit_sex

def RestingECG_val(col):
    if col == 'Normal':
        return 0.0
    elif col == 'ST':
        return 0.75
    else:
        return 1.0

RestingECG_udf = udf(lambda val: RestingECG_val(val), DoubleType())
df_after_edit_RestingECG = final_df.withColumn('RestingECG', RestingECG_udf('RestingECG'))

df_after_edit_RestingECG.show()

def ST_Slope_val(col):
    if col == 'Flat':
        return 0.0
    elif col == 'Up':
        return 0.75
    else:
        return 1.0

ST_Slope_udf = udf(lambda val: ST_Slope_val(val), DoubleType())
df_teliko = df_after_edit_RestingECG.withColumn('ST_Slope', ST_Slope_udf('ST_Slope'))

df_teliko.show()

df_teliko.groupBy("sex").count().show()

df_teliko.groupBy("Oldpeak").count().show()

df_teliko.groupBy("FastingBS").count().show()

df_teliko.groupBy("ExerciseAngina").count().show()

df_teliko.groupBy("HeartDisease").count().show()

### converting the spark df to a pandas df in order to perform transformations 

pandasDF = df_teliko.toPandas()

print(pandasDF)

###############################################################  Creating noise ###########################################################################

# mu is the mean of the normal distribution we are choosing from
# std is the standard deviation

mu=0.0
std = 0.1
def gaussian_noise(pandasDF,mu,std):
    noise = np.random.normal(mu, std, size = pandasDF.shape)
    x_noisy = pandasDF + noise
    return x_noisy

###############################################################  adding noise ###########################################################################
import numpy as np

x_noisy = pandasDF + np.random.normal(mu, std, size = pandasDF.shape)

print(x_noisy)

import pandas as pd
import numpy as np

x_noisy['HeartDisease'] = np.where(x_noisy['HeartDisease'] > 0.7 ,1,0) 
#pd.cut(x_noisy['HeartDisease'], bins=[0, 0.45, 0.55, 10], include_lowest=True, labels=['0', 'NaN', '1'])

x_noisy.groupby(['HeartDisease']).size()

print(x_noisy)

# times  st age
x_noisy.groupby(['Age']).size()

## categorize sex values below or above 0.5
 
x_noisy['sex'] = np.where(x_noisy['sex'] >= 0.5 ,1,0)

x_noisy.groupby(['sex']).size()

#### The rows where sex=1 and sex=0 diddnt change a lot so i perform some random changes in these values  
#### Randomly change the 5% of 1s to 0s

v = x_noisy.sex.values == 1
x_noisy.loc[v, 'sex'] = np.random.choice((0, 1), v.sum(), p=(.05, .95))

x_noisy.groupby(['sex']).size()

print(x_noisy)

# replace negative numbers with NaN
#x_noisy[x_noisy < 0] = "NaN"

## we could consider some negative values as outliners in the dataframe 

#x_noisy[x_noisy < 0] = 0

### FastingBS

x_noisy.groupby(['FastingBS']).size()

### values vary from -0.32 to 1.26
### those that are < -0.30 and > 1.2 can e considers as outliners 

#x_noisy['FastingBS'] = pd.cut(x_noisy['FastingBS'], bins=[-0.326525 , -0.25, 0.65, 0.85, 1.264831], include_lowest=True, labels=['0', '0.5', '0.75','1'])
x_noisy['FastingBS'] = pd.cut(x_noisy['FastingBS'], bins=[-0.455938, -0.254442, -0.250807, -0.234021, -0.224653, -0.204652, 0.5, 1.190075, 1.203776, 1.220578, 1.242796, 1.259308, 1.495296], include_lowest=True, labels=['-0.326524', '-0.300144', '-0.287943', '-0.284272', '-0.279712', '0.0', '1.0', '1.213249', '1.215685', '1.226302', '1.228461', '1.264830'])

x_noisy.groupby(['FastingBS']).size()


### ExerciseAngina
x_noisy.groupby(['ExerciseAngina']).size()

x_noisy['ExerciseAngina'] = pd.cut(x_noisy['ExerciseAngina'], bins=[-0.455938, -0.354442, -0.300807, -0.254021, -0.224653, -0.204652, 0.5, 1.200075, 1.203776, 1.220578, 1.242796, 1.259308, 1.495296], include_lowest=True, labels=['-0.316524', '-0.290144', '-0.287943', '-0.284272', '-0.279712', '0.0', '1.0', '1.213249', '1.215685', '1.226302', '1.228461', '1.264830'])

x_noisy.groupby(['ExerciseAngina']).size()

print(x_noisy)

# FastingBS: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# ExerciseAngina exercise induced angina (1 = yes; 0 = no)

# cant have an age of 43,25 so we round the value to 43                  
cols = ['Age']                  
x_noisy[cols] = x_noisy[cols].applymap(np.int64)

print(x_noisy)

#### ChestPainType
x_noisy.groupby(['ChestPainType']).size()

x_noisy['ChestPainType'] = pd.cut(x_noisy['ChestPainType'], bins=[-0.45, -0.20, -0.19, 0.35, 0.65, 0.85, 1.15, 1.20, 1.40 ], include_lowest=True, labels=['-0.30', '-0.20','0', '0.5', '0.75','1', '1.19', '1.39'])

x_noisy.groupby(['ChestPainType']).size()

print(x_noisy)

### RestingECG
x_noisy.groupby(['RestingECG']).size()

x_noisy['RestingECG'] = pd.cut(x_noisy['RestingECG'], bins=[-0.40,-0.30, 0.55, 0.85, 1.25, 1.50], include_lowest=True, labels=['-0.35','0', '0.75','1','1.4'])

x_noisy.groupby(['RestingECG']).size()

print(x_noisy)

### ST_Slope
x_noisy.groupby(['ST_Slope']).size()

x_noisy['ST_Slope'] = pd.cut(x_noisy['ST_Slope'], bins=[-0.40, -0.25, 0.55, 0.85, 1.15, 1.29, 1.50], include_lowest=True, labels=['-40','0', '0.75','1','1.29','1.50'])

x_noisy.groupby(['ST_Slope']).size()

#### default oldpeak column values vary from -0,26 to 6.2 

x_noisy.groupby(['Oldpeak']).size()

print(x_noisy)

######################################  adding random NaN times ##################################################################################

cols = ['Age', 'sex', 'ChestPainType', 'Cholesterol', 'ExerciseAngina']                  

# frac=0.1 --> 10% of every column 
for cols in x_noisy[cols]:
    x_noisy.loc[x_noisy.sample(frac=0.1).index, cols] = pd.np.nan

##################### for better allocation in some collumns we add 10% NaN values while in some other we add 7% NaN values

cols2 = ['RestingBP', 'FastingBS', 'RestingECG', 'MaxHR', 'Oldpeak', 'ST_Slope']                  

for cols2 in x_noisy[cols2]:
    x_noisy.loc[x_noisy.sample(frac=0.07).index, cols2] = pd.np.nan

count_nan = len(x_noisy) - x_noisy.count()
count_nan

print(x_noisy)

x_noisy.dtypes

######## final dataframe with noise, outliers and NaN values ###########################

print(x_noisy)

###### type conversion #######

## convert into numeric values

x_noisy = x_noisy.apply(pd.to_numeric, errors='ignore')

print(x_noisy)

###### final 
## data types 
x_noisy.to_csv('out.csv')


### upload on github --> https://github.com/gkolovos/kaggleDATAset

x_noisy.dtypes

print(x_noisy)
