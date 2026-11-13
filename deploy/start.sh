#!/bin/sh
# Run the API and the UI in one container. The host routes public traffic to $PORT (the UI);
# the UI talks to the API on localhost.
set -e
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
export API_URL=http://127.0.0.1:8000
exec streamlit run ui/streamlit_app.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
