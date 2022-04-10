FROM python:3.8

WORKDIR /ApnaLeadsPredictor

COPY ./requirements.txt .

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8087"]