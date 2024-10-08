#!/bin/bash

# Start Spark worker
export SPARK_MASTER_HOST=10.228.233.5  # Update this to your master's hostname or IP
export SPARK_MASTER_PORT=7077
export SPARK_WORKER_WEBUI_PORT=8081

$SPARK_HOME/sbin/start-worker.sh spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT

echo "Spark worker started and connected to master at $SPARK_MASTER_HOST:$SPARK_MASTER_PORT"
echo "Worker Web UI available at http://$(hostname):$SPARK_WORKER_WEBUI_PORT"
