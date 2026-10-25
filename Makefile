up:
	docker compose up -d

build:
	docker compose build

ingest:
	docker compose exec app python -m src.ingest

logs:
	docker compose logs -f app

down:
	docker compose down

local:
	streamlit run ui/app.py
