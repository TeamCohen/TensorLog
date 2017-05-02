import simple
b = simple.RuleBuilder()
p,q = b.predicates("p,q")
person_t,place_t = b.types("person_t,place_t")
w = p(person_t,place_t) & q(person_t)
b.schema += w
