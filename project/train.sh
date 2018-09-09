#!/bin/sh
export SERVICE_HOME="$(cd "`dirname "$0"`"/..; pwd)"

# define your environment variable
export JAVA_HOME="/Library/Java/JavaVirtualMachines/jdk1.8.0_162.jdk/Contents/Home"
export SPARK_HOME='/usr/local/spark-2.3.0-bin-hadoop2.6'

spark-submit ${SERVICE_HOME}/project/train.py ${SERVICE_HOME}
