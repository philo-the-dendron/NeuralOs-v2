# NeuralOS v2 paper — build, figures, gates.
# Figures python: repo-local venv if present (python3 -m venv .figvenv &&
# .figvenv/bin/pip install matplotlib), else system python3.
# Fixed epoch: matplotlib clamps PDF CreationDate to it, so `make figs`
# regenerates byte-stably (P2-W3; verified: two runs, identical bytes).
export SOURCE_DATE_EPOCH ?= 1700000000
FIGPYTHON ?= $(shell if [ -x $(PWD)/.figvenv/bin/python ]; then echo $(PWD)/.figvenv/bin/python; else echo python3; fi)
TEX := $(shell if command -v tectonic >/dev/null; then echo tectonic; \
        elif command -v latexmk >/dev/null; then echo latexmk -pdf; \
        elif command -v pdflatex >/dev/null; then echo pdflatex; \
        else echo nix run nixpkgs#tectonic --; fi)

FIGS = figs/basins.pdf figs/ladder.pdf figs/adjudication_table.pdf \
	figs/mechanism.pdf figs/flip_cis.tex

.PHONY: all build figs gate clean dist

all: build

build: $(FIGS)
	$(TEX) main.tex

figs: $(FIGS)

figs/%.pdf: figs/%.py
	cd figs && $(FIGPYTHON) $(notdir $<)

figs/%.tex: figs/%.py
	cd figs && $(FIGPYTHON) $(notdir $<)

gate:
	sh tools/check_language.sh

# Zenodo bundles (Z1). Paper zip = gated PDF + full source + MANIFEST,
# named by the paper base commit. Repo artifact = git archive of the
# published-crate commit (103fa59, neuralos-snn 0.1.0-alpha.5 tree).
# dist/ is gitignored; sha256s land in dist/SHA256SUMS and the ISA.
BASE_SHA := $(shell git rev-parse --short=7 HEAD)
REPO_SHA := 103fa59
PAPER_ZIP := dist/neuralos-branchb-paper-$(BASE_SHA).zip
REPO_TGZ := dist/neuralos-repo-$(REPO_SHA).tar.gz

dist: build figs gate
	rm -rf dist && mkdir -p dist/stage/sections dist/stage/figs dist/stage/tools
	cp main.tex Makefile refs.bib MANIFEST.md dist/stage/
	cp main.pdf dist/stage/main.pdf
	cp sections/*.tex dist/stage/sections/
	cp figs/*.py figs/*.pdf figs/*.tex figs/README.md dist/stage/figs/
	cp tools/check_language.sh dist/stage/tools/
	cd dist/stage && zip -X -q -r ../$(notdir $(PAPER_ZIP)) .
	git -C .. archive --format=tar.gz -o $(CURDIR)/$(REPO_TGZ) $(REPO_SHA)
	rm -rf dist/stage
	cd dist && sha256sum $(notdir $(PAPER_ZIP)) $(notdir $(REPO_TGZ)) > SHA256SUMS
	@cd dist && ls -la $(notdir $(PAPER_ZIP)) $(notdir $(REPO_TGZ)) && cat SHA256SUMS

clean:
	rm -f main.pdf main.aux main.log main.out main.bbl main.blg main.fls main.fdb_latexmk
	rm -f figs/*.pdf figs/flip_cis.tex
