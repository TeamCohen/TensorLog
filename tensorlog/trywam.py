# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import parser
import wamcompiler
import waminterpreter
import factplugin

if __name__ == "__main__":

    rules = parser.RuleCollection()
    rules.add(parser.Parser().parseRule('p(X,Y) :- spouse(X,Y) {r}.'))
    rules.listing()
    wp = wamcompiler.Compiler().compileRules(rules)
    wp.listing()
    fp = factplugin.FactPlugin.load('../test/fam.cfacts')
    wi = waminterpreter.Interpreter(wp,plugins=[fp])
    print wi.plugins
    query = parser.Parser().parseQuery('p(X,Y).')
    print query
    answers = waminterpreter.Util.answer(wi,query)
    print answers
