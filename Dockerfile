FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
