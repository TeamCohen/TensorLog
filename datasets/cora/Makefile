expt:
	python cora-expt.py

#cora.db: cora.cfacts
#	time python $(TLOG)/matrixdb.py --serialize cora.cfacts cora.db
#

cora.cfacts: 
	cat raw/cora-*.cfacts > cora.cfacts

clean:
	rm -rf *~ 
	rm -rf cora-trained.db cora.db 




