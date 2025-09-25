BUILD_DIR = build
TEX_FILE = text_to_image_no_text

.PHONY: pdf clean all

all: pdf

pdf:
	mkdir -p $(BUILD_DIR)
	pdflatex -output-directory=$(BUILD_DIR) $(TEX_FILE).tex
	bibtex $(BUILD_DIR)/$(TEX_FILE)
	pdflatex -output-directory=$(BUILD_DIR) $(TEX_FILE).tex
	pdflatex -output-directory=$(BUILD_DIR) $(TEX_FILE).tex

quick:
	mkdir -p $(BUILD_DIR)
	pdflatex -output-directory=$(BUILD_DIR) $(TEX_FILE).tex

clean:
	rm -rf $(BUILD_DIR)

view: pdf
	open $(BUILD_DIR)/$(TEX_FILE).pdf

help:
	@echo "Available targets:"
	@echo "  pdf     - Build the complete PDF with bibliography (default)"
	@echo "  quick   - Quick build (single pdflatex run, no bibliography)"
	@echo "  clean   - Remove build directory"
	@echo "  view    - Build and open the PDF"
	@echo "  help    - Show this help"