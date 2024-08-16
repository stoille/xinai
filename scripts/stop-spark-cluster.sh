#!/bin/bash

# Stop all Spark processes
export SPARK_HOME=/path/to/spark  # Update this path

$SPARK_HOME/sbin/stop-master.sh
$SPARK_HOME/sbin/stop-slave.sh

echo "All Spark processes have been stopped"
