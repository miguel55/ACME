#!/bin/bash

#Generating the PDF with pdflatex
pdflatex -synctex=1 -interaction=nonstopmode doc.tex
bibtex doc
pdflatex -synctex=1 -interaction=nonstopmode doc.tex
pdflatex -synctex=1 -interaction=nonstopmode doc.tex
rm {*.aux,,*.log,*.out,*.synctex.gz,*toc,*blg,*bbl}
