#!/bin/bash
conda activate faiss_gpu1
streamlit run /media/udonsi-kalu/New\ Volume/denials/denials/app.py \
  --server.port=8502 \
  --server.baseUrlPath=app \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.headless=true