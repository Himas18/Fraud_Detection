FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "-c", "uvicorn backend:app --host 0.0.0.0 --port=8000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
