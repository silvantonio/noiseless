FROM python:2.7.8
MAINTAINER Antonio Silva "a.silva@pointerbp.nl"

ENV PROJECT_PATH /code

RUN mkdir -p $PROJECT_PATH
ADD requirements.txt $PROJECT_PATH/
WORKDIR $PROJECT_PATH
RUN pip install -r requirements.txt
ADD . $PROJECT_PATH

#RUN apt-get update && apt-get install -y python3-tk
#RUN python -m nltk.downloader brown

CMD ["python", "app.py"]


