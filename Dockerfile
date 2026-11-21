# Week 9: single image for the Streamlit app, the scheduler and the one-shot migrate job.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libpq for psycopg2-binary is bundled; build tools only needed if a wheel is missing
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image ~1 GB smaller than the default wheel
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src ./src
COPY app ./app
COPY models ./models
COPY configs ./configs
COPY analysis ./analysis
COPY data ./data
COPY docs/week4/baseline_metrics.csv ./docs/week4/baseline_metrics.csv
COPY docs/week6/final_comparison.csv ./docs/week6/final_comparison.csv
COPY docs/week7/results_summary.json ./docs/week7/results_summary.json
COPY docs/week9/perf_baseline.json ./docs/week9/perf_baseline.json
COPY docs/week10/panel_summary.json ./docs/week10/panel_summary.json
COPY docs/week11/ml_round2_summary.json ./docs/week11/ml_round2_summary.json
COPY docs/week12 ./docs/week12

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
