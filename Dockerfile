FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY dashboard/ dashboard/
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/main_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]