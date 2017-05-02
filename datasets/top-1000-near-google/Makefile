# default experiment and its output

expt: expt.log eval.log

expt.log:
	python top-1000-near-google.py > expt.log

eval.log: expt.log
	rm -f $@
	for f in tmp-cache/*.solutions.txt; do \
	echo file $$f >> eval.log; \
	proppr eval tmp-cache/top-1000-near-google-test.examples $$f --metric auc --defaultNeg >> $@; \
	done

# a quickly-viewable check on the default experiment (like a unit test)

check: actual.txt
	diff -y actual.txt expected.txt || true

unittest:
	python testexpt.py 

actual.txt: expt.log eval.log
	echo \# actual output of expt on `date` > actual.txt
	grep training.*done expt.log >> actual.txt
	grep -e file -e micro eval.log >> actual.txt 

# prepare for running expt
setup: 
	mkdir -p tmp-cache

# clean up the directory

clean:
	rm -rf *~ *.log actual.txt
	rm -rf tmp-cache/*.solutions.txt tmp-cache/*.examples
