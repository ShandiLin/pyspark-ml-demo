#!/usr/bin/python
import traceback


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


def predict_data(spark, logger, model_path, data):
    '''
        preprocess without pipeline
    '''
    from pyspark.ml.regression import RandomForestRegressionModel
    from project.schema import get_pred_schema
    try:
        assert len(data) > 0, 'empty data'
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
        return None


def predict_data_pipe(spark, logger, model_path, data):
    '''
        preprocess with pipeline
    '''
    from pyspark.ml import PipelineModel
    from project.pyspark_pipeline_wrapper import PysparkPipelineWrapper
    from project.schema import get_pred_schema
    try:
        assert len(data) > 0, 'empty data'
        logger.info("{} rows".format(len(data)))

        # create spark dataframe
        spark_data = spark.createDataFrame(data, get_pred_schema())

        # load and predict
        model = PysparkPipelineWrapper.unwrap(PipelineModel.load(model_path))
        pred = model.transform(spark_data)
        return [p['prediction'] for p in pred.collect()]

    except Exception:
        logger.error(traceback.print_exc())
        return None
