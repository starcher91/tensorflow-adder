FROM python:2

RUN pip install tensorflow

RUN mkdir -p /usr/local/app
COPY main.py /usr/local/app/main.py
WORKDIR /usr/local/app

ENV TF_CPP_MIN_LOG_LEVEL='2'

CMD ["python", "main.py"]