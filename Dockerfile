FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

COPY ./app/ /code/app/
COPY ./main.py /code/main.py

EXPOSE 5000  

CMD ["python", "main.py"]
