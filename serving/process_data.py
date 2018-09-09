#!/usr/bin/python


def preprocess(df):
    import pyspark.sql.functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from project.preprocess import str2num, get_strIdxers

    df = df.withColumn('OverTime', str2num(F.col('OverTime'), {'No': 0, 'Yes': 1})) \
        .withColumn('Gender', str2num(F.col('Gender'), {'Male': 0, 'Female': 1})) \
        .withColumn('MaritalStatus', str2num(F.col('MaritalStatus'),
                                             {'Married': 2, 'Divorced': 1, 'Single': 0})) \
        .withColumn('BusinessTravel', str2num(F.col('BusinessTravel'),
                                              {'Travel_Frequently': 2, 'Non-Travel': 0, 'Travel_Rarely': 1})) \
        .drop('Over18')

    strIdx_cols = ["Department", "EducationField", "JobRole"]
    pipeline = Pipeline(stages=get_strIdxers(strIdx_cols))
    df = pipeline.fit(df).transform(df) \
        .drop(*strIdx_cols)

    assembler = VectorAssembler(inputCols=df.columns, outputCol="features")
    vector_df = assembler.transform(df).select('features')
    return vector_df


def predict_data(model_path, data):
    import traceback
    from pyspark.ml.regression import RandomForestRegressionModel
    from project.spark import get_spark, get_logger
    from project.schema import get_pred_schema

    try:
        assert len(data) > 0, 'empty data'
        spark = get_spark(app_name="pred")
        logger = get_logger(spark, "pred_logger")
        logger.info("{} rows".format(len(data)))

        # create spark dataframe
        spark_data = spark.createDataFrame(data, get_pred_schema())

        # preprocessing
        preprocess_data = preprocess(spark_data)

        # load and predict
        m = RandomForestRegressionModel.load(model_path)
        pred = m.transform(preprocess_data)
        return [p['prediction'] for p in pred.collect()]

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()
