.PHONY: report clean preprocess-all embed db-setup db-load faiss experiments proxy proxy-setup run

# Load environment variables from .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Set default Cloud SQL instance if not provided
CLOUD_SQL_INSTANCE ?= pristine-flames-460002-h2:us-west1:postgres

# Variables
NOTEBOOK = notebook/generate_figures.ipynb
REPORT_FILE = report/final/capstone_final_report.qmd

# Train indices and load data
train: csv embed db-load faiss

# Individual preprocessing targets
csv:
	python src/preprocess/clean_data.py

embed:
	python src/preprocess/generate_embed.py

db-load:
	python src/preprocess/load_db.py

faiss:
	python src/preprocess/compute_faiss_index.py

# Run experiments and track results on MLflow
experiments:
	python experiments/experiment_pipeline.py

# Render the Quarto document
report:
	@echo "Executing notebook to generate charts..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)
	@echo "Rendering Quarto report..."
	quarto render $(REPORT_FILE)

# Cloud SQL Proxy commands
proxy-setup:
	chmod +x scripts/setup_sql_proxy.sh
	./scripts/setup_sql_proxy.sh

proxy:
	@if [ ! -f .cloud_sql_proxy/cloud_sql_proxy ]; then \
		echo "Proxy not set up. Running setup first..."; \
		$(MAKE) proxy-setup; \
	else \
		./.cloud_sql_proxy/cloud_sql_proxy -instances="$(CLOUD_SQL_INSTANCE)"=tcp:5433; \
	fi

# Start both frontend and backend applications
run:
	$(MAKE) proxy & \
	python src/backend/api.py & \
	sleep 2 && \
	streamlit run src/frontend/app.py & \
	wait

# Clean up temporary files and shutdown applications
clean:
	@echo "Cleaning up..."
	@-pkill -f "streamlit run src/frontend/app.py" || true
	@-pkill -f "python src/backend/api.py" || true
	@-pkill -f "cloud_sql_proxy" || true
	rm -f report/final/capstone_final_report.pdf
	rm -rf report/final/capstone_final_report_files
	rm -f img/faiss_hyperparam.png
	rm -f img/recall_chart.png
	rm -f img/precision_chart.png
	rm -f img/search_time_chart.png
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .pytest_cache
	rm -rf **/.pytest_cache
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	@echo "Unsetting environment variables..."
	@grep -v '^\s*#' .env | grep '=' | cut -d '=' -f 1 | xargs -I {} sh -c 'unset {}; echo "Unset {}";' || true