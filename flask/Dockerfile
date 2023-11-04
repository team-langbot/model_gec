FROM ubuntu:latest

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
         gcc \
         python3-dev \
         curl \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py

COPY ./requirements.txt /opt/code/requirements.txt

RUN pip install -r /opt/code/requirements.txt

# RUN pip install pandas numpy boto3 flask torch simpletransformers && rm -rf /root/.cache

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV FLASK_APP=/opt/code/gec_inference.py

# Set up the program in the image
COPY gec_cows_l2h_small.pkl /opt/code/gec_cows_l2h_small.pkl
COPY gec_inference.py /opt/code/gec_inference.py
WORKDIR /opt/code

ENTRYPOINT flask run --host=0.0.0.0
# RUN flask run