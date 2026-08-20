# NeuralOS v2 paper — build, figures, gates.
# Figures python: repo-local venv if present (python3 -m venv .figvenv &&
# .figvenv/bin/pip install matplotlib), else system python3.
FIGPYTHON ?= $(shell if [ -x ./.figvenv/bin/python ]; then echo ./.figvenv/bin/python; else echo python3; fi)
TEX := $(shell if command -v tectonic >/dev/null; then echo tectonic; \
        elif command -v latexmk >/dev/null; then echo latexmk -pdf; \
        elif command -v pdflatex >/dev/null; then echo pdflatex; \
        else echo nix run nixpkgs#tectonic --; fi)

FIGS = figs/basins.pdf figs/ladder.pdf figs/adjudication_table.pdf figs/mechanism.pdf

.PHONY: all build figs gate clean

all: build

build: $(FIGS)
	$(TEX) main.tex

figs: $(FIGS)

figs/%.pdf: figs/%.py
	cd figs && $(FIGPYTHON) $(notdir $<)

gate:
	sh tools/check_language.sh

clean:
	rm -f main.pdf main.aux main.log main.out main.bbl main.blg main.fls main.fdb_latexmk
	rm -f figs/*.pdf
