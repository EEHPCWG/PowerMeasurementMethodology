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
	pdflatex methodology | tee latex.out ; \
	pdflatex methodology | tee latex.out; \
	pdflatex methodology | tee latex.out

clean:
	find . -name '*.blg' -print0 | xargs -0 rm -f; \
	find . -name '*.aux' -print0 | xargs -0 rm -f; \
	find . -name '*.bbl' -print0 | xargs -0 rm -f; \
	find . -name '*.log' -print0 | xargs -0 rm -f; \
	find . -name '*.out' -print0 | xargs -0 rm -f; \
	find . -name '*.toc' -print0 | xargs -0 rm -f; \
	find . -name '*.lof' -print0 | xargs -0 rm -f; \
	find . -name '*.lot' -print0 | xargs -0 rm -f; \
	find . -name '*.fdb_latexmk' -print0 | xargs -0 rm -f; \
	find . -name '*.fls' -print0 | xargs -0 rm -f; \
	rm -f methodology.pdf
