# FROM python:3.8-buster
FROM python:3.11-slim-bookworm

# RUN apt-get install gcc
# ADD odbcinst.ini /etc/odbcinst.ini
# RUN apt-get update
# RUN apt-get install -y tdsodbc unixodbc-dev
# RUN apt install unixodbc-bin -y
# RUN apt-get clean -y

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . .
CMD streamlit run --server.port 8501 --server.enableCORS false Realtor_Buddy.py



