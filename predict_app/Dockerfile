FROM python:3.11
WORKDIR /app
COPY . /app/
RUN pip install --upgrade pip
RUN apt-get -y install libc-dev
RUN pip install -r requirements.txt
CMD [ "python", "main.py" ]