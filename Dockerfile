FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY src ./src
COPY docs ./docs
COPY reports ./reports
COPY data ./data
COPY .env.example ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD python -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
