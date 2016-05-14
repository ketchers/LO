#13.6.3 Use the binomial theorem to find a single term 

import tools
import sympy as sym
import random
from sympy.functions.combinatorial.numbers import nC, nP


class FindTermInBinomial():
    """
    This class has a single static method stem() that generates the problems.
    """

    def __init__(self):
        # Cache for problems already created
        self.done = []
        self.count = 0
        print self.count

    def stem(self, a_type = None, max_size = 10000, choose_style = None):
        """
        This generates problems asking the students to find the coefficent on the x^n*y^m
        term in the expansion of (ax + by)^p where m + n = p.
        
        Named Parameters
            a_type       -- This determines if the answer is multiple choice (MC) or free response (FR)
            max_size     -- Don't accept problems whose answer is biigger in magnitude than this.
            choose_style -- choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_%s C_%s'], if choose_style
                            is < len(choose_styles), then it chooses the associated style, else it chooses randomly
        """
        
        #Make a copy of current args for later use.
        
        kwargs = {'a_type': a_type,
                  'max_size': max_size,
                  'choose_style': choose_style}
      
        
        # If an answer is biger than this generate a new problem
        THREASH = 85000 
        
        x, y = sym.symbols('x, y')

        self.count += 1
        
        if self.count > 200:
            print "too many iterations"
            return None

        if a_type == None:
            a_type = random.choice(["MC", "FR"])
            

        # Choose some coefficients a, b for (ax + by)^p
        coeffs = range(-7,8)
        coeffs.remove(0)
        a, b = [random.choice(coeffs) for i in range(2)]
        # Choose a power p in [5,8]
        power = random.randint(4, 8)
        x_pow = random.randint(2, power - 2)
        y_pow = power - x_pow
        

        if (a_type, a, b, x_pow, y_pow, power) in self.done:
            return self.stem(**kwargs)
        
        self.done.append((a_type, a, b, x_pow, y_pow))

        
        question_stem = "Find the coefficient of $_x^{%s}y^{%s}$_ in the expansion of \
        $_(%s)^{%s}$_." % (x_pow, y_pow, sym.latex(sym.simplify(a*x+b*y)), power)

        answer = nC(power, x_pow) * a**x_pow * b**y_pow

        if abs(answer) > THREASH:
            return self.stem(**kwargs)

        # Chose the "choose" --- Oh have to be a little tricky here since at one place
        # we have choose(n,k) and another we have choose(%s,$s)%(power,x_pow) 
        
        choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_%s C_%s']
        
        if choose_style not in range(len(choose_styles)):
            choose_ = random.choice(choose_styles)
        else:
            choose_ = choose_styles[choose_style]
            
        def choose(a ,b):
            choice =  choose_ % (a, b)
            return choice
        
        explanation = "According to the Binomial Theorem," \
                      "$$(a x + b y)^n = \\sum_{k = 0}^{n} %s \\cdot (ax)^k(by)^{n - k}$$" \
                      "In this case, $_n = %d$_, $_a = %d$_, $_b = %d$_, and $_k = %d$_, " \
                      "so the coefficient is" % (choose('n','k'), power, a, b, x_pow)
        explanation += tools.align("%s \\cdot (%s)^{%s} \\cdot (%s)^{%s}"
                                   % (choose(power,x_pow), a, x_pow, b, y_pow),
                                   "\\frac{%s!}{%s!\\,%s!} \\cdot (%s)^{%s} \\cdot (%s)^{%s}"
                                   % (power, x_pow, power, a, x_pow, b, y_pow),
                                   sym.latex(answer))

        if a_type == "MC":
            errors = list(set([nC(power + i, x_pow + i // 2)*a**x_pow*b**y_pow for i in range(-3,4) if i != 0 and x_pow + i // 2 > 0]))
            errors += [int(-answer*(1.02)), int(answer*(1.02)), int(-answer*(0.98)), int(answer*(0.98))]
            errors = [e for e in errors if e != answer]
            random.shuffle(errors)
            errors = errors[:4] # This provides 4 distractions


        if a_type == "FR":
            question_stem += " Give your answer as an integer."
            answer_mathml = tools.fraction_mml(answer)
            return tools.fully_formatted_question(question_stem, explanation, answer_mathml)
        else:
            distractors = [answer] + errors
            distractors = ["$_%s$_" % sym.latex(distractor) for distractor in distractors]
            return tools.fully_formatted_question(question_stem, explanation, answer_choices=distractors)
        # stem
        
if __name__ == "__main__":
    
    prob = FindTermInBinomial()
   
    for i in range(5):
        print prob.stem(a_type = 'MC')
        
    
    # You could check the prob.done cache
    # print prob.done
    
    for i in range(5):
        print prob.stem(a_type = 'FR', choose_style = 2)
        
    # You could check the prob.done cache (this will extend what you see above)
    # print prob.done
  
    