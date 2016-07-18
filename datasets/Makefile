# conventions for the subdirectories 
#   - (cd foo; make) should run some sort of experiment, which
#       should take < 5min to run
#   - (cd foo; make check) should display actual-vs-expected results 
#   - (cd foo; make clean) should cleanup so that 'git status .' sees nothing unexpected
#
# hence 'make clean test check' should redo all the experiments and the tail
# will be inspectable for obvious errors
#
#   - foo/inputs holds inputs directly used for the experiments
#   - foo/raw, if present, holds stuff used to populate foo/inputs
#   - foo/tmp-cache holds outputs and/or intermediate results


test:
	(cd cora; make)
	(cd wordnet; make)
	(cd grid; make)
	(cd fb15k-237; make)
	(cd smokers; make)
#	(cd textcat-ssl; make)

check:
	(cd cora; make check)
	(cd wordnet; make check)
	(cd grid; make check)
	(cd fb15k-237; make check)
	(cd smokers; make check)
#	(cd textcat-ssl; make check)

clean:
	(cd cora; make clean)
	(cd wordnet; make clean)
	(cd grid; make clean)
	(cd fb15k-237; make clean)
	(cd smokers; make clean)
#	(cd textcat-ssl; make clean)
