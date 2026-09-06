FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential for packages that compile
# native extensions, e.g. xgboost/shap wheels on some platforms). git is
# not needed: pyopdb is vendored locally at data/pyopdb/, not pip-installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
