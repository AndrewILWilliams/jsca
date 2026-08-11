# Minimal makefile for building the jsca documentation locally.
#
#   pip install -r requirements.txt   # once
#   make html                         # build into _build/html
#   make clean                        # remove the build tree

SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
