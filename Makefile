.PHONY: report clean index embed db-setup db-load faiss experiments proxy proxy-setup run

# Load environment variables from .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

#
# Cloud SQL Proxy commands
#
# Set default Cloud SQL instance if not provided
CLOUD_SQL_INSTANCE ?= pristine-flames-460002-h2:us-west1:postgres

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

#
# Indexing commands
#
index: csv embed db-load faiss

# Individual indexing targets
csv:
	python src/indexing/clean_data.py

embed:
	python src/indexing/generate_embed.py

db-load:
	python src/indexing/load_db.py

faiss:
	python src/indexing/compute_faiss_index.py

#
# MLFlow experiment commands
#
experiments:
	python experiments/experiment_pipeline.py

#
# Quarto report building commands
#
NOTEBOOK = report/final/generate_figures.ipynb
REPORT_FILE = report/final/capstone_final_report.qmd

report:
	@echo "Executing notebook to generate charts..."
	jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)
	@echo "Rendering Quarto report..."
	quarto render $(REPORT_FILE)
	mv _site/report/final/capstone_final_report.pdf report/final/capstone_final_report.pdf

#
# Start both frontend and backend applications
#
run:
	$(MAKE) proxy & \
	python src/backend/api.py & \
	sleep 2 && \
	streamlit run src/frontend/app.py & \
	wait

#
# Clean up temporary files and shutdown applications
#
clean:
	@echo "Cleaning up..."
	@-pkill -f "streamlit run src/frontend/app.py" || true
	@-pkill -f "python src/backend/api.py" || true
	@-pkill -f "cloud_sql_proxy" || true
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
	rm -rf _site
	@echo "Unsetting environment variables..."
	@grep -v '^\s*#' .env | grep '=' | cut -d '=' -f 1 | xargs -I {} sh -c 'unset {}; echo "Unset {}";' || true