from __future__ import division
from __future__ import print_function
import tools
import random
import datetime
import sympy as sym


class ExpandBinomial():
    """
    This will generate problems on expanding binomials.
    """
    
    def __init__(self, seed=None):
        # Cache for problems already created
        self.done = set()
        self.count = 0
        if seed is None:
            self.seed = datetime.datetime.now().microsecond
        else:
            self.seed = seed
        random.seed(seed)  # Set  once per instance
        
    def stem(self, a_type=None, choose_style=0):
        """
        Parameters:
        ----------
        a_type       : ['MC','FR']. The answer type. Multple choice or free response. 
        choose_style : (integer index into choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_%s C_%s']). 
                           If choose_style is < len(choose_styles), then it chooses the associated style, 
                           else it chooses randomly
        """
   
        kwargs = {
            'a_type': a_type,
            'choose_style': choose_style
        }
    
        choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_%s C_%s']
        
        if choose_style not in range(len(choose_styles)):
            choose_ = random.choice(choose_styles)
        else:
            choose_ = choose_styles[choose_style]
            
        def choose(a , b):
            choice = choose_ % (a, b)
            return choice
    
        THREASH = 10000  # Restrict to coeffs smaller than this
        LET = 'x' * 4 + 'y' * 4 + 'z' * 3 + 'a' * 2 + 'b' * 2 + 'f' * 2 + '1' * 6
        
        x = random.choice(LET)
        y = random.choice([z for z in LET if z != x])
     
        if a_type is None:
            a_type = random.choice(['MC', 'FR'])
            
        if a_type == 'MC':
            exp_choices = sum([[3] * 4, [4] * 3, [5] * 2], [])
            coef_choices = range(-6, 7)
            coef_choices.remove(0)
        else:
            exp_choices = sum([[3] * 4, [4] * 3], [])
            coef_choices = range(-5, 6)
            coef_choices.remove(0)
        
        a = random.choice(coef_choices)
        b = random.choice(coef_choices)
        n = random.choice(exp_choices)
        
        expr = self.gen_prob(a, b, n, x, y)
        
        if expr in self.done or self.get_max_coeff(expr) > THREASH:
            return self.stem(**kwargs)
        self.done.add(expr)
        
        explanation = 'Recall that ' + tools.align(sym.latex(expr),
                '\\sum_{i=0}^{%s}(%s)^{(%s - i)}\\,(%s)^i\\,%s\\,%s'\
                    % (n, a, n, b, choose(n, 'i'),
                       sym.latex(sym.sympify('%s**(%s-i)*%s**i' % (x, n, y)))),
                sym.latex(sym.expand(expr)))
            
        answer = sym.expand(expr)

        
        if a_type == 'MC':
            errors = self.gen_errs(a, b, n, x, y)
            errors = [er for er in errors if self.get_max_coeff(er) < THREASH]
            question_stem = 'Which of the given choices is the correct expansion of $$%s$$' % (sym.latex(expr))
            distractors = [answer] + errors[0:4]
            distractors = ["$_%s$_" % sym.latex(distractor) for distractor in distractors]
            return tools.fully_formatted_question(question_stem, explanation, answer_choices=distractors)
        else:
            question_stem = 'Give the expansion of $$%s$$.' % (sym.latex(expr))
            answer_mathml = tools.itex2mml("$_" + sym.latex(answer) + "$_")
            return tools.fully_formatted_question(question_stem, explanation, answer_mathml)
            
    
    def gen_prob(self, a, b, n, x, y):
        return sym.sympify('(%s*%s + %s*%s)**%s' % (a, x, b, y, n))
    
    def gen_errs(self, a, b, n, x, y):
        """
        What is the best thing to do here?
        """
        errs = set()
        
        # Generate som obvious candidates
        errs.add(sym.expand(self.gen_prob(-a, -b, n, x, y)))
        if a < 0:
            errs.add(sym.expand(self.gen_prob(-a, b, n, x, y)))
        if b < 0:
            errs.add(sym.expand(self.gen_prob(a, -b, n, x, y)))
       
        
        expr = sym.expand(self.gen_prob(a, b, n, x, y))
        coeffs = sym.Poly(expr).coeffs()
        
        for i in range(4):
            
            expr1 = self.gen_poly(map(lambda l: random.choice([-1, 1]) * l, coeffs), x, y)
            expr2 = self.gen_poly(map(lambda l: random.choice([-2, -1, 0, 1, 2]) * l, coeffs), x, y)
            errs.update([expr1, expr2, expr - (expr2 - sym.LM(expr))])
        
        errs = list(errs)
        errs_ = [err for err in errs if err != expr]
        random.shuffle(errs_)
        return errs
        
    
    def gen_poly(self, ls, x, y):
        """
        Parameters:
        ----------
        ls  : list of coefficients length n + 1
        x,y : symbols or strings 
        """
        n = len(ls)
        return sum([sym.sympify('%s*%s**%s*%s**%s' % (ls[i], x, (n - i - 1), y, i)) 
                            for i in range(len(ls))], sym.sympify(0))
        
    
    def get_max_coeff(self, e):
        """
        Gets the maximal coefficient from (ax+by)**n given either as 
        a string or sympy expr.
        """
        if type(e) is str:
            return self.get_max_coeff(sym.sympify(e))
        ep = sym.Poly(e)
        return max(map(sym.Abs, ep.coeffs())) 
    
    
if __name__ == "__main__":
    prob = ExpandBinomial(seed=43)
    
    ex = ""
    
    for i in range(10):
        ex += "<p class=\'problem\'>"
        ex += prob.stem(a_type='FR')
        ex += "</p>\n"
        
 
    for i in range(10):
        ex += "<p class='problem\'>"
        ex += prob.stem(a_type='WC', choose_style=1)
        ex += "</p>\n"

    
    for i in range(10):
        ex += "<p class=\'problem\'>"
        ex += prob.stem(choose_style=None)
        ex += "</p>\n"
    
    print(ex)
