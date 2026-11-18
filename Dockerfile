# Container for the Hugging Face Space (Docker SDK). Data is downloaded at startup from a
# private Hugging Face dataset (see deploy/fetch_data.py); secrets come from Space settings.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WORKDIR was created by root; the app runs as `user` and must be able to write data/ (database + usage.db)
RUN mkdir -p /home/user/app/data && chown -R user:user /home/user/app
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    TRUST_PROXY=1 \
    PYTHONUNBUFFERED=1

# Bake the query-time search models into the image so cold starts don't download them.
RUN python -c "from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer; \
[AutoTokenizer.from_pretrained(m) for m in ('ncbi/MedCPT-Query-Encoder', 'ncbi/MedCPT-Cross-Encoder')]; \
AutoModel.from_pretrained('ncbi/MedCPT-Query-Encoder'); \
AutoModelForSequenceClassification.from_pretrained('ncbi/MedCPT-Cross-Encoder')"

COPY --chown=user app ./app
COPY --chown=user scripts ./scripts
COPY --chown=user deploy ./deploy

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
