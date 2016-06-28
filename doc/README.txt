
HOW TO RUN AN EXPERIMENT:

Look at the sample main in src/expt.py, and the sample input files in
src/test/textcattoy.cfacts and src/test/textcat.ppr.  Some other
larger examples are in datasets/cora/cora-expt.py and
datasets/wordnet/wnet-expt.py.


HOW TO CONFIGURE TENSORLOG:

Some of the modules have a config.Config() object, which is just an
object that contains fields which can be used as options.  Any
user-settable parameters should be in these objects.

HOW TO SERIALIZE A .cfacts FILE AND CREATE A DB FILE:

  % python matrixdb.py --serialize foo.cfacts foo.db

HOW TO DEBUG A TENSORLOG PROGRAM:

Start up an interpreter with the command

  % python -i -m tensorlog --programFiles foo.db:foo.ppr:foo.cfacts:...

You can then evaluate functions with commands like:

 % ti.eval("foo/io", "input_constant")

Try setting these config options before you do 

  ops.trace = True
  conf.trace = True

You can also insert "printf literals" into a clause, eg

  p(X,Z1):-printf(X,X1),spouse(X1,Y),printf(Y,Y1),sister(Y1,Z),printf(Z,Z1).

These literals just copy the input to the output, but will echo the
bindings of the variables when the message-passing happens.  (Make
sure the output variable of the printf is used somewhere "downstream",
otherwise it's undefined when the print will actually happen.)

