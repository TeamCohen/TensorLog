default:
	echo make what?

cora.db: cora.cfacts
	time python $(TLOG)/matrixdb.py --serialize cora.cfacts cora.db

cora.cfacts: 
	cat cora-*.cfacts > cora.cfacts

