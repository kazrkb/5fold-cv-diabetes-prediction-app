FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
