# (C) William W. Cohen and Carnegie Mellon University, 2016
#
# version number tracking for Tensorlog

VERSION = '1.3.4'

# externally visible changes:
#
# version 1.0: refactored cleaned-up version of nips-submission codebase
# version 1.1: thread-safe gradient and eval computation, and a parallel learner
# version 1.1.1: cleaned up trace function api, bug fix in parallel learner
# version 1.1.2: tracer output is not per example, no parallel option in funs
# version 1.2.0: not sure, really.
# version 1.2.1: plearn replaces epoch-level status monitoring with merged results minibatches
# version 1.2.2: add learner option to expt command-line
# version 1.2.3:
#    add p(X,Y) :- ... {foo(F): q(X,F)} templates, propprProg.setRuleWeights(), propprProg.setFeatureWeights()
#    list --prog xxx --ruleids
#    more options for expt
# version 1.2.4:
#    added --params and --weightEpsilon to expt.py
#    made conf.ops.long_trace a number
#    added interp.set()
# version 1.2.5:
#    cross-compilation
# version 1.3.0:
#    tensorlog is a module
#    new api for cross compilers + "simple" api
#    type declaration in cfacts: # :- actedIn(actor_t,movie_t)
#    parameter declarations:     # :- trainable(posWeight,1)
#    OOV marker for test/train .exam files
#    interp.Interp split off from program
# version 1.3.1:
#     simple.Compiler() fleshed out and tested for tensorflow
# version 1.3.1a:
#     AbstractCrossCompiler.possibleOps() added
# version 1.3.2:
#     binary user-defined plugins, eg
#       plugins = program.Plugins()
#       plugins.define('double/io', lambda x:2*x, lambda inputType:inputType)
#       prog = program.ProPPRProgram(rules=rules,db=db,plugins=plugins)
#     simple.RuleBuilder
# version 1.3.3:
#     split of version.py into different file
#     refactored schema
#     simple.RuleBuilder -> simple.Builder
# version 1.3.4:
#     bug fix in type inference
#     new serialization and use of file-like objects for load* methods
#       dbschema.serializeTo(filelike)
#       db.serializeDataTo(filelike,filter=None|params|noparams)
