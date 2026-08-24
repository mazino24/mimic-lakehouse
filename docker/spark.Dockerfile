# Spark master/worker image.
#
# Based on the official Apache image rather than bitnami/spark: Bitnami
# withdrew their public Docker Hub catalogue, so `bitnami/spark:3.5.3` no
# longer resolves and every build fails at the FROM line.
#
# Adds the jars the base image does not ship: S3A (MinIO) and Postgres JDBC.
FROM apache/spark:3.5.3-python3

USER root

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262
ARG POSTGRES_JDBC_VERSION=42.7.3
ARG MAVEN=https://repo1.maven.org/maven2

RUN apt-get update \
 && apt-get install --no-install-recommends -y curl ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o ${SPARK_HOME}/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
      ${MAVEN}/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar \
 && curl -fsSL -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
      ${MAVEN}/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
 && curl -fsSL -o ${SPARK_HOME}/jars/postgresql-${POSTGRES_JDBC_VERSION}.jar \
      ${MAVEN}/org/postgresql/postgresql/${POSTGRES_JDBC_VERSION}/postgresql-${POSTGRES_JDBC_VERSION}.jar \
 && chmod a+r ${SPARK_HOME}/jars/*.jar

# The lakehouse package must be importable on executors as well as the driver.
ENV PYTHONPATH=/opt/spark-apps
WORKDIR /opt/spark-apps

# uid 185 is the `spark` user in the official image.
USER 185
