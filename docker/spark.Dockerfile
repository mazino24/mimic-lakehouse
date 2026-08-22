# Spark master/worker image: Spark 3.5 plus the jars needed to talk to MinIO
# (S3A) and Postgres (JDBC), which the base image does not ship.
FROM bitnami/spark:3.5.3

USER root

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262
ARG POSTGRES_JDBC_VERSION=42.7.3
ARG MAVEN=https://repo1.maven.org/maven2

RUN install_packages curl ca-certificates \
 && curl -fsSL -o /opt/bitnami/spark/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
      ${MAVEN}/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
 && curl -fsSL -o /opt/bitnami/spark/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
      ${MAVEN}/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
 && curl -fsSL -o /opt/bitnami/spark/jars/postgresql-${POSTGRES_JDBC_VERSION}.jar \
      ${MAVEN}/org/postgresql/postgresql/${POSTGRES_JDBC_VERSION}/postgresql-${POSTGRES_JDBC_VERSION}.jar

# The lakehouse package must be importable on executors as well as the driver.
ENV PYTHONPATH=/opt/spark-apps
WORKDIR /opt/spark-apps

USER 1001
