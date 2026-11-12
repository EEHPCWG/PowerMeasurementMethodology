$pdflatex = 'pdflatex -8bit -etex -halt-on-error -synctex=1 %O %S';
$pdf_mode = 1;
$bibtex_use = 1;
$clean_ext .= ' cut run.xml synctex.gz';
$clean_full_ext .= ' bbl';
