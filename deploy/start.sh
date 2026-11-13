#!/bin/sh
# Run the API and the UI in one container. Only the UI is public (on $PORT); the API listens on
# localhost inside the container, so it cannot be called directly from the internet.
set -e
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
export API_URL=http://127.0.0.1:8000
exec streamlit run ui/streamlit_app.py \
  --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true \
  --server.maxUploadSize 10 --browser.gatherUsageStats false
