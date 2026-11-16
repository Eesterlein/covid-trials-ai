#!/bin/sh
set -e
python -m deploy.fetch_data
exec uvicorn app.server:app --host 0.0.0.0 --port 7860
