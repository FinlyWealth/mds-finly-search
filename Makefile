.PHONY: report clean preprocess-all generate-embed load-db compute-faiss

# Default target
all: preprocess-all report

# Combined preprocessing target
preprocess-all: generate-embed load-db compute-faiss

# Individual preprocessing targets
generate-embed:
	python preprocess/generate_embed.py

load-db:
	python preprocess/load_db.py

compute-faiss:
	python preprocess/compute_faiss_index.py

# Render the Quarto document
report: report/capstone_proposal_report.qmd
	quarto render report/capstone_proposal_report.qmd

# Clean up temporary files
clean:
	rm -f report/capstone_proposal_report.pdf
	rm -rf report/capstone_proposal_report_files