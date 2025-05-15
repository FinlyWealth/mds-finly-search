.PHONY: report clean preprocess-all embed db-setup db-load faiss experiments proxy proxy-setup run

# Default target
all: proxy-setup run

# Combined preprocessing target
preprocess-all: embed db faiss

# Individual preprocessing targets
embed:
	python preprocess/generate_embed.py

db-setup:
	chmod +x scripts/setup_local_db.sh
	./scripts/setup_local_db.sh

db-load:
	python preprocess/load_db.py

faiss:
	python preprocess/compute_faiss_index.py

# Run experiments and track results on MLflow
experiments:
	python experiments/experiment_pipeline.py

# Render the Quarto document
report: report/capstone_proposal_report.qmd
	quarto render report/capstone_proposal_report.qmd

# Cloud SQL Proxy commands
proxy-setup:
	chmod +x scripts/setup_sql_proxy.sh
	./scripts/setup_sql_proxy.sh

proxy:
	@if [ ! -f .cloud_sql_proxy/cloud_sql_proxy ]; then \
		echo "Proxy not set up. Running setup first..."; \
		$(MAKE) proxy-setup; \
	else \
		./.cloud_sql_proxy/cloud_sql_proxy -instances="regal-campus-456918-d2:us-west1:finly-mds"=tcp:5433; \
	fi

# Start both frontend and backend applications
run:
	streamlit run src/frontend/ap.py & \
	python src/backend/api.py & \
	wait

# Clean up temporary files
clean:
	rm -f report/capstone_proposal_report.pdf
	rm -rf report/capstone_proposal_report_files
	rm -rf .cloud_sql_proxy