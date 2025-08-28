#!/bin/bash
conda activate faiss_gpu1
streamlit run /media/udonsi-kalu/New\ Volume/denials/denials/streamlit_app.py \
  --server.port=8503 \
  --server.baseUrlPath=agent \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.headless=true