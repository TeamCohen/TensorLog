expt: inputs/cora.cfacts
	python cora-exptv2.py


setup: inputs/cora.cfacts

#cora.db: cora.cfacts
#	time python $(TLOG)/matrixdb.py --serialize cora.cfacts cora.db
#

inputs/cora.cfacts: 
	cat raw/cora-*.cfacts > $@

clean:
	rm -rf *~ 
	rm -rf cora-trained.db cora.db 




