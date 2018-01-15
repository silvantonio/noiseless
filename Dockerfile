FROM python:2.7.8
MAINTAINER Antonio Silva "a.silva@pointerbp.nl"

ENV PROJECT_PATH /code
ENV CORPORA all #brown

RUN mkdir -p $PROJECT_PATH
ADD . $PROJECT_PATH
WORKDIR /code

RUN pip install -r requirements.txt \
    && apt-get update \
    && apt-get install -y python3-tk

RUN python -m nltk.downloader $CORPORA
CMD ["python", "app.py"]


