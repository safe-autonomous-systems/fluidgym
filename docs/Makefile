.PHONY: clean html linkcheck docs

clean:
	rm -rf build
	rm -rf source/api
	find docs -name "*.rst" -path "*generated*" -delete

html:
	SPHINX_GALLERY_PLOT=False sphinx-build -b html source build/html -E
	@echo
	@echo "Build finished. The HTML pages are in build/html."

linkcheck:
	SPHINX_GALLERY_PLOT=False sphinx-build -b linkcheck build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."

docs: clean html linkcheck
