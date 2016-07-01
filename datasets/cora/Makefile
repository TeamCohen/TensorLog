expt: setup
	python cora-expt.py

setup: inputs/cora.cfacts

inputs/cora.cfacts: 
	cat raw/cora-*.cfacts > $@

clean:
	rm -rf *~ 
	rm -rf cora-trained.db cora.db 




