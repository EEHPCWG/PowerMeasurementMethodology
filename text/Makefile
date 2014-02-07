# Makefile but currently more like a batch file
# Issue a make clean and then a make
#

TARGETS: methodology.pdf

.PHONY: all clean complete methodology

all: methodology.pdf
methodology: methodology.pdf

tex_files = $(wildcard *.tex)
bib_files = $(wildcard */*.bib)
pdf_files = $(wildcard *.pdf)

methodology.pdf: $(tex_files) $(bib_files) $(pdf_files)
	pdflatex -shell-escape methodology | tee latex.out ; \
	pdflatex -shell-escape methodology | tee latex.out; \
	pdflatex -shell-escape methodology | tee latex.out; \
	cp methodology.pdf ../downloadPDF/methodology.pdf

clean:
	find . -name '*.blg' -print | xargs rm -f; \
	find . -name '*.aux' -print | xargs rm -f; \
	find . -name '*.bbl' -print | xargs rm -f; \
	find . -name '*.log' -print | xargs rm -f; \
	find . -name '*.out' -print | xargs rm -f; \
	rm -f methodology.pdf

