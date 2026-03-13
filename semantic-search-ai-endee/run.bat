@echo off
echo Make sure Endee is running: docker compose up -d
echo Make sure you've run: python ingest.py
echo.
uvicorn api:app --reload --port 8000
