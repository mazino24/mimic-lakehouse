# Airflow image: scheduler/webserver plus a local Spark client, so
# SparkSubmitOperator can submit to the standalone cluster in client mode.
FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update \
 && apt-get install --no-install-recommends -y openjdk-17-jre-headless procps curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
# The image is built for both arm64 and amd64; pick whichever JVM is present.
RUN if [ ! -d "$JAVA_HOME" ]; then \
      echo "JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))" >> /etc/environment; \
    fi

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262
ARG POSTGRES_JDBC_VERSION=42.7.3
ARG MAVEN=https://repo1.maven.org/maven2

RUN mkdir -p /opt/spark-jars \
 && curl -fsSL -o /opt/spark-jars/hadoop-aws.jar \
      ${MAVEN}/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
 && curl -fsSL -o /opt/spark-jars/aws-java-sdk-bundle.jar \
      ${MAVEN}/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
 && curl -fsSL -o /opt/spark-jars/postgresql.jar \
      ${MAVEN}/org/postgresql/postgresql/${POSTGRES_JDBC_VERSION}/postgresql-${POSTGRES_JDBC_VERSION}.jar \
 && chmod -R a+rX /opt/spark-jars

USER airflow

COPY requirements/airflow.txt /tmp/airflow-requirements.txt
RUN pip install --no-cache-dir -r /tmp/airflow-requirements.txt

ENV PYTHONPATH=/opt/spark-apps:/opt/airflow/dags
