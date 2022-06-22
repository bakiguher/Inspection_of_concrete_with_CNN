FROM python:3
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /code/
CMD ["python", "app.py"]
EXPOSE 5000
