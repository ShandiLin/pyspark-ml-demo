import os
import sys
import shutil
import traceback

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from spark import get_spark, get_logger
from preprocess import run_preprocess
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

        # preprocess
        vector_df = run_preprocess(df)

        # seperate train and valid
        logger.info('preprocessing')
        (train_data, valid_data) = vector_df.randomSplit([0.8, 0.2])

        # training
        logger.info('training')
        rf = RandomForestRegressor(
            labelCol="label", featuresCol="features", numTrees=10)
        model = rf.fit(train_data)

        # get validation metric
        predictions = model.transform(valid_data)
        evaluator = RegressionEvaluator(
            labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        logger.info('valid rmse: {}'.format(rmse))

        # save or update model
        model_path = SERVICE_HOME + '/model'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info('model exist, rm old model')
        model.save(model_path)
        logger.info('save model to {}'.format(model_path))

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()
