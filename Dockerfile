FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11
RUN apt-get update && apt-get install -y cmake
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY  . .