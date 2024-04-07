# Start from the Spark base image
FROM docker.io/bitnami/spark:3.5

# Install dependencies
USER root
RUN mkdir -p /var/lib/apt/lists/partial
RUN apt-get update -y && apt-get clean
RUN apt-get upgrade -y && apt-get -y install tree
RUN apt-get install -y python3-pip
RUN apt-get install -y git && apt-get install -y git-lfs && git lfs install

RUN mkdir -p /spark_pipeline && cd /spark_pipeline

COPY requirements.txt /spark_pipeline/requirements.txt

# Install Python dependencies
RUN pip install -r /spark_pipeline/requirements.txt


COPY postgresql-42.7.2.jar /spark_pipeline/postgresql-42.7.2.jar
COPY ./hugging_face/git-clone.sh /spark_pipeline/git-clone.sh
COPY ./hugging_face/model_repo_urls.txt /spark_pipeline/model_repo_urls.txt
COPY *.py / 

RUN chmod +r /spark_pipeline/postgresql-42.7.2.jar

RUN /bin/bash /spark_pipeline/git-clone.sh

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=4041", "--no-browser", "--allow-root", "--NotebookApp.token=''" ,"--NotebookApp.password=''" ]
