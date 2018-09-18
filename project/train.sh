#!/bin/sh
export SERVICE_HOME="$(cd "`dirname "$0"`"/..; pwd)"

# define your environment variable
export SPARK_HOME=/opt/spark-2.3.1-bin-hadoop2.6
export PATH=$PATH:$SPARK_HOME/bin
export JAVA_HOME=/usr/lib/jvm/java
export PYTHONPATH=$PATH:$SERVICE_HOME}/project:$SERVICE_HOME}/serving

spark-submit ${SERVICE_HOME}/project/train.py ${SERVICE_HOME}
