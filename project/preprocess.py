import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


def str2num(v, map_dict):
    def str2num_(v):
        return map_dict.get(v)
    return F.udf(str2num_, IntegerType())(v)


def get_strIdxers(cols):
    return [StringIndexer(inputCol=c, outputCol=c + '_Idx') for c in cols]


def run_preprocess(df):
    df = df.withColumn('OverTime', str2num(F.col('OverTime'), {'No': 0, 'Yes': 1})) \
        .withColumn('Attrition', str2num(F.col('Attrition'), {'No': 0, 'Yes': 1})) \
        .withColumn('Gender', str2num(F.col('Gender'), {'Male': 0, 'Female': 1})) \
        .withColumn('MaritalStatus', str2num(F.col('MaritalStatus'),
                                             {'Single': 0,
                                              'Divorced': 1,
                                              'Married': 2})) \
        .withColumn('BusinessTravel', str2num(F.col('BusinessTravel'),
                                              {'Non-Travel': 0,
                                               'Travel_Rarely': 1,
                                               'Travel_Frequently': 2})) \
        .withColumnRenamed('Attrition', 'label') \
        .drop('Over18')

    strIdx_cols = ["Department", "EducationField", "JobRole"]
    pipeline = Pipeline(stages=get_strIdxers(strIdx_cols))
    df = pipeline.fit(df).transform(df) \
        .drop(*strIdx_cols)

    features = [c for c in df.columns if c != 'label']
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    vector_df = assembler.transform(df).select(*['features', 'label'])
    return vector_df
