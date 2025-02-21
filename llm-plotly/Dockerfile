FROM python:3.9-slim-buster
LABEL maintainer="Shuen.Lyu <shuen.lyu@uct.com>"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /LLM
WORKDIR /LLM

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8007
CMD ["python", "llm.py"]
