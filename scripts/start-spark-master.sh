#!/bin/bash

# Start Spark master
export SPARK_HOME=/opt/spark  # Update this path
export SPARK_MASTER_HOST=$(hostname)
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080

$SPARK_HOME/sbin/start-master.sh

echo "Spark master started on $SPARK_MASTER_HOST:$SPARK_MASTER_PORT"
echo "Web UI available at http://$SPARK_MASTER_HOST:$SPARK_MASTER_WEBUI_PORT"
