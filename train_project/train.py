import os
import sys
import shutil
import traceback

import pyspark.sql.functions as F
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from myspark import get_spark, get_logger
from preprocess import str2num
from pipeline import get_stages, create_pipeline
from schema import get_train_schema

assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'


def main():

    try:
        SERVICE_HOME = sys.argv[1]

        # init spark
        spark = get_spark(app_name="sample")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        df = spark.read.schema(get_train_schema()).option('header', True).csv(
            SERVICE_HOME + '/dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

        # label preprocessing (only in training part)
        df = df.withColumn('label', str2num(
            F.col('Attrition'), {'No': 0, 'Yes': 1})) \
            .drop('Attrition')

        # seperate train and valid
        (train_data, valid_data) = df.randomSplit([0.8, 0.2])

        # preprocess(pipeline / non-pipeline) / training
        logger.info('preprocessing & training')
        stages = get_stages(train_data)
        rf = RandomForestRegressor(
            labelCol="label", featuresCol="features", numTrees=10)
        stages.append(rf)
        mypipeline = create_pipeline(stages)
        mymodel = mypipeline.fit(train_data)

        # get validation metric
        predictions = mymodel.transform(valid_data)
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        logger.info('valid rmse: {}'.format(rmse))

        model_path = SERVICE_HOME + '/model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info('model exist, rm old model')
        mymodel.save(model_path)
        logger.info('save model to {}'.format(model_path))

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()
