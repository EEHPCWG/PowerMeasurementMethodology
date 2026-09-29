MAINFILE := methodology.tex

default: build

build:
	latexmk -pdf -synctex=1 ${MAINFILE}

open: build
	latexmk -pv -view=pdf ${MAINFILE}

clean:
	latexmk -pdf -c

distclean:
	latexmk -pdf -C

.PHONY: default build open clean distclean
