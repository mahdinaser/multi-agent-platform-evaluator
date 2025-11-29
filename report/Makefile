# Makefile for IEEE Conference Paper

# Main target
TARGET = paper

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compile the paper
all: $(TARGET).pdf

# Build PDF
$(TARGET).pdf: $(TARGET).tex
	@echo "Compiling LaTeX document..."
	$(LATEX) $(TARGET).tex
	@echo "Running BibTeX..."
	-$(BIBTEX) $(TARGET)
	@echo "Recompiling for references..."
	$(LATEX) $(TARGET).tex
	$(LATEX) $(TARGET).tex
	@echo "Done! Output: $(TARGET).pdf"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	@echo "Clean complete!"

# Clean everything including PDF
distclean: clean
	@echo "Removing PDF output..."
	rm -f $(TARGET).pdf
	@echo "Distribution clean complete!"

# Quick compile (no bibtex)
quick:
	@echo "Quick compile (no bibliography update)..."
	$(LATEX) $(TARGET).tex
	@echo "Quick compile complete!"

# View PDF (cross-platform)
view: $(TARGET).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(TARGET).pdf; \
	elif command -v open > /dev/null; then \
		open $(TARGET).pdf; \
	elif command -v start > /dev/null; then \
		start $(TARGET).pdf; \
	else \
		echo "Could not find PDF viewer. Please open $(TARGET).pdf manually."; \
	fi

.PHONY: all clean distclean quick view

