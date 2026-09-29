MAIN  = rapport
BUILD = build

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex bib/bibM2.bib cls/llncs.cls
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
	        -outdir=$(BUILD) $(MAIN)
	cp $(BUILD)/$(MAIN).pdf .

clean:
	latexmk -C -outdir=$(BUILD)
	rm -f $(MAIN).pdf

.PHONY: all clean