import sys as sys
import tools
import sympy as sym
import random
import itertools as iter
import distutils
from sympy.functions.combinatorial.numbers import nC, nP

#13.1.3a Evaluate an expression with factorials
#13.6.1 Find Binomial Coefficient use stem(q_type = "C", include_PC = True)

class FactCombPerm():
    """
    Evaluate some simple expressions involving factorials.
    """
    def __init__(self):
        
        self.done = []
        
    @staticmethod
    def format_fact(n, m = 1):
        """
        Formats the expansion of a factorial nicely.
        
        Parameters:
            n -- The largest value
            m -- The value at which to stop
        
        If n - m s large just write the first three terms ... and then m, e.g., 
        fromat_fact(20, 5) gives "20 \cdot 19 \cdot 18 \cdots 5" 
        """
        strg = ""
        if n - m > 5:
            strg = "%s \\cdot %s \\cdot %s \\cdots %s" % (n, n - 1, n - 2, m)
        else:
            k = n
            while k > m:
                strg = strg + str(k) + " \\cdot "
                k -= 1
            strg = strg + str(m)
    
        return strg # FactCombPerm.format_fact
    
    def stem(self, q_type = None, a_type = None, include_PC = False, perm_style = None, choose_style = None):
        """
        There are three possible types of problems: F = factorials, e.g. 5!, P = 
        permutations, e.g. 5!/3! = (5 - 2)!, and C = combinations, e.g. 5!/(3!2!).
        
        Named Parameters:
        
            q_type     -- "Fact", "Perm", or "Comb" for factorial, permutation, or combination
            a_type     -- "MC" or "FR" for multiplechoice and free response
            include_PC -- This is a boolean. If true include mention of permutations/combinations
                          in both problems and explanations.
            perm_style -- perm_styles = ['P(%s,%s)', '{}_%s P_%s'], if choose_style
                          is < len(choose_styles), then it chooses the associated style, 
                          else it chooses randomly
                          
            choose_style -- choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_%s C_%s'], if choose_style
                            is < len(choose_styles), then it chooses the associated style, else it 
                            chooses randomly             
        """
        
        # Assume if the user sets the perm_style or choose_style, assume include_PC should be True
        if perm_style is not None or choose_style is not None:
            include_PC = True
        
        kwargs = {
                    'q_type': q_type,
                    'a_type': a_type,
                    'include_PC': include_PC,
                    'perm_style': perm_style,
                    'choose_style': choose_style
                 }
        
        # These numbers can get big lets set a threashold
        THREASH = 300000
        
        # If we exceed the threashold restart with current settings
        ## I wish I had a better wa to get current values???
        def check():
            if answer > THREASH:
                self.stem(**kwargs)
            else: 
                pass
        
        q_type_orig = q_type
        a_type_orig = a_type
        
        # About 1/5 simple factorials, 2/5 perms, 3/5 combinations
        if q_type == None:
            q_type = random.choice(["Fact"] + ["Perm"] * 2 + ["Comb"] * 3)
        
        if a_type == None:
            a_type = random.choice(["MC", "FR"])
                
        # Actually set the data for the problem
        if q_type == "Fact":
            N = random.randint(5,7) # Change these to modify range
            R = 0
        else:
            N = random.randint(7,11) # Change these to modify range
            R = random.randint(3, N - 4) # To make the explanations work it is good to have N - R > 3
            
            
        
        # Choose the "choose" --- Oh have to be a little tricky here since at one place
        # we have choose(n,k) and another we have choose(%s,$s)%(power,x_pow) 
        
        choose_styles = ['{%s \\choose %s}', 'C(%s,%s)', '{}_{%s} C_{%s}']
        perm_styles = ['P(%s,%s)', '{}_{%s} P_{%s}']
        
        if choose_style not in range(len(choose_styles)):
            choose_ = random.choice(choose_styles)
        else:
            choose_ = choose_styles[choose_style]
            
            
        # Now choose the "perm"
        
        if perm_style not in range(len(perm_styles)):
            perm_ = random.choice(perm_styles)
        else:
            perm_ = perm_styles[perm_style]
            
        
        def choose(a ,b):
            if q_type == "Comb":
                choice =  choose_ % (a, b)
            else:
                choice =  perm_ % (a, b)
            return choice
        
         
        
        
        q_data = (q_type, N, R) # This determines the  problem
        
        if q_data in self.done:
            return self.stem(q_type_orig, a_type_orig, include_PC) # Try again
        else:
            self.done.append(q_data)  # Mark this data as being used
            
       
        # If include_PC is True, this will be overriden below.
        question_stem_options = ["Evaluate the following expression involving factorials."]
        question_stem = random.choice(question_stem_options)
        
        explanation = "Recall the definition: "
        
        
        if q_type == "Fact":
            
            answer = sym.factorial(N)
            check()
            
            if a_type == "MC":
                errors = [sym.factorial(N+1), sym.factorial(N-1), 
                          sym.factorial(N)/sym.factorial(random.randint(1,N-1))]
            
            question_stem += "$$%s!$$" % (N)
            
            explanation += "$_%s! = %s = %s$_" % (N, FactCombPerm.format_fact(N), answer)
                                         
        elif q_type == "Perm":
            
            if include_PC == True:
                question_stem = "Evaluate the following permutation."
            
            answer = nP(N, N - R)
            check()
            
            
            if a_type == "MC":
                
                errors = list(set([nP(N + i, N - (R + i)) for i in range(-3,4) if i != 0]))
                errors = [e for e in errors if e != answer]
                random.shuffle(errors)
                errors = errors[:4] # This provides 4 distractions
                
                
            
            if include_PC:
                explanation_prefix = "%s = \\frac{%s!}{(%s - %s)!} =" % (choose(N, N - R), N, N, N - R)
            else:
                explanation_prefix = ""
        
            explanation += "$_%s\\frac{%s!}{%s!} = \\frac{%s}{%s} =  %s = %s$_" \
            % (explanation_prefix, N, R, FactCombPerm.format_fact(N), 
               FactCombPerm.format_fact(R), FactCombPerm.format_fact(N, R+1), answer)
          
            if include_PC:
                question_stem += "$$%s$$" %(choose(N, N - R))
            else:
                question_stem += "$$\\frac{%s!}{%s!}$$" % (N, R)
            
        else:
            
            if include_PC == True:
                question_stem = "Evaluate the following combination."
                
            answer = nC(N, R) 
            check()
            
            if a_type == "MC":
                errors = list(set([nC(N + i, R + i // 2) for i in range(-3,4) if i != 0 and R + i // 2 > 0]))
                errors = [e for e in errors if e != answer]
                random.shuffle(errors)
                errors = errors[:4] # This provides 4 distractions
                
                
            if include_PC:
                explanation_prefix = "%s = \\frac{%s!}{%s!\\,(%s - %s)!} = " % (choose(N, R), N, R, N, R)
            else:
                explanation_prefix = ""
        
          
            if R >= N - R:
                explanation += tools.align("%s\\frac{%s!}{%s!\,%s!}" \
                                           % (explanation_prefix, N, R, N-R),
                                           "\\frac{%s}{(%s)(%s)}" \
                                           % (FactCombPerm.format_fact(N), FactCombPerm.format_fact(R), 
                                              FactCombPerm.format_fact(N-R)), 
                                           "\\frac{%s}{%s} = %s" \
                                           % (FactCombPerm.format_fact(N, R + 1), FactCombPerm.format_fact(N - R), answer))
            else:
                explanation += tools.align("%s\\frac{%s!}{%s!\,%s!}" \
                                           % (explanation_prefix, N, R, N-R),
                                           "\\frac{%s}{(%s)(%s)}" \
                                           % (FactCombPerm.format_fact(N), FactCombPerm.format_fact(R),
                                              FactCombPerm.format_fact(N-R)), 
                                           "\\frac{%s}{%s} = %s" \
                                           % (FactCombPerm.format_fact(N, (N - R) + 1), FactCombPerm.format_fact(R), answer))
                
                
            
            if include_PC:
                question_stem += "$$%s$$" % (choose(N, R))
            else:
                question_stem += "$$\\frac{%s!}{%s!\\,%s!}$$" % (N, R, N - R)
        
            explanation += "<br>"                           
    
    
    
        if a_type == "FR":
            question_stem += "Give your answer as an integer."
            answer_mathml = tools.fraction_mml(answer)
            return tools.fully_formatted_question(question_stem, explanation, answer_mathml)
        else:
            distractors = [answer] + errors
            distractors = ["$_%s$_" % sym.latex(distractor) for distractor in distractors]
            return tools.fully_formatted_question(question_stem, explanation, answer_choices=distractors)
        # stem
   
    
    

if __name__ == "__main__":
    
    prob = FactCombPerm()
    
    for i in range(5):
        print prob.stem(q_type = "Perm", perm_style = 1, a_type = 'MC')
        
    
        
    for i in range(5):
        print prob.stem(q_type = "Comb", choose_style = 1, a_type = 'FR')
        
    
    for i in range(10):
        print prob.stem() # Just random, no perm /comb
