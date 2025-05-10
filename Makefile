.PHONY: report clean preprocess-all embed db faiss

# Default target
all: preprocess-all report

# Combined preprocessing target
preprocess-all: embed db faiss

# Individual preprocessing targets
embed:
	python preprocess/generate_embed.py

db:
	python preprocess/load_db.py

faiss:
	python preprocess/compute_faiss_index.py

# Render the Quarto document
report: report/capstone_proposal_report.qmd
	quarto render report/capstone_proposal_report.qmd

# Clean up temporary files
clean:
	rm -f report/capstone_proposal_report.pdf
	rm -rf report/capstone_proposal_report_files