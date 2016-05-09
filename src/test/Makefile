

VPATH=../

fb15k-valid.db: matrixdb.py
	python $< --serialize fb15k-valid.cfacts:fb15k.tensorLog.cfacts $@

fb15k: tensorlog.py fb15k-valid.db 
	python -i $< --programFiles fb15k-valid.db:fb15k.tensorLog.ppr --proppr --debug

