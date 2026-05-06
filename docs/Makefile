.PHONY: clean html landing-page docs

clean:
	rm -rf build

html:
	SPHINX_GALLERY_PLOT=False sphinx-build -b html source build/html/docs -E
	@echo
	@echo "Build finished. The HTML pages are in build/html."

landing-page:
	cp landing_page/index.html build/html/index.html
	cp -r landing_page/static build/html

docs: clean html landing-page
