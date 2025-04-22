## Makefile for building the Sphinx documentation
# You can also run sphinx-build directly: sphinx-build -b html . _build/html

# Directory for source files
SOURCEDIR = .
# Directory for build output
BUILDDIR  = _build

.PHONY: help clean html

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html      to build the HTML documentation"

html:
	@sphinx-build -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	@rm -rf $(BUILDDIR)
