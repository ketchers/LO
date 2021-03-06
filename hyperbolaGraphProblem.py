from __future__ import division
from __future__ import print_function
import sys
sys.path.append('finished')
import tools
import os
from urllib import quote_plus
import random
import sympy as sym
import numpy as np
import matplotlib as mpl
import pylab as plt
from R_tools import make_func
from html_tools import html_image, make_table
from hyperbola import Hyperbola
from table import Table


x, y, z = sym.symbols('x,y,z')

class HyperbolaGraphProblem(object):
    
    def __init__(self, seed=None, path='hyperbola_plots'):
        """
        As usual done is a list that keeps track of what problems have been generated.
        """
        self.done = set()
        self.count = 0

        self.path = path

        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            
    def gen_errors(self, hyp, xkcd = False, force = False):
        #Put in the hyperbola that is just like the current one with
        # major/minor axes swapped.
        if hyp.trans == 'x':
            errors = [Hyperbola(a = hyp.a, b = hyp.b, trans='y')]
        else:
            errors = [Hyperbola(a = hyp.a, b = hyp.b, trans='x')]
            
        while len(errors) < 6:
            errors.append(Hyperbola(a = range(max(1,hyp.a - 2), hyp.a + 3), 
                                b = range(max(1,hyp.b - 2), hyp.b + 3),
                                trans = ['x','y'],
                                h = range(hyp.h - 2, hyp.h + 2), 
                                k = range(hyp.k - 2, hyp.k + 2)))
        random.shuffle(errors)
        errors = [er for er in errors if er != hyp]
        errors = errors[0:4]
        #The following generates the error images.
        errors = [er.show(path = self.path, force = force, xkcd = xkcd) 
                  for er in errors]
        return errors
    
    def stem(self, a_type = 'preview', xkcd = False, force = False, 
             expanded = True, fmt = 'latex'):
        """
        Parameters:
        ----------
        a_type   : String ('preview', 'MC')
            Is this for testing / preview? If so set to true.
        expanded : Boolean
            If true, the equation is fully expanded and the student must
            get the equation into normal form.
        """
        
        kwargs = {
            'a_type':a_type
        }
        
        prob = Hyperbola()
        
        if prob in self.done:
            return self.stem(**kwargs)
        else:
            self.done.add(prob)
            self.count += 1
            
        if a_type == 'preview':
            question_stem = "<br><h2>Question</h2><br>"
        else:
            question_stem = ""
        
        if expanded:
            const = (prob.a**2) * (prob.b **2)
            question_stem += 'Find the normal form of the hyperbola given by the \
                equation $$%s = 0$$ and use this \
            to select the correct graph of the given hyperbola.' \
            % ( sym.latex(const * (prob.expr.expand() - 1)))
        else:
            question_stem += 'Select the correct graph of the hyperbola given\
            by the equations: $$%s = 1$$' % (prob.latex)
        
        ans = prob.show(path=self.path, xkcd = xkcd, force = force)
        
        if a_type == 'preview':
            explanation = "<br><h2>Explanation</h2><br>"
        else:
            explanation = ""
                                                         
        explanation += prob.explanation(path=self.path + "/explanation", 
                                   a_type = a_type, fmt = fmt, 
                                   expanded = expanded,
                                   xkcd = xkcd)
                                                         
        errors = self.gen_errors(prob, force = force, xkcd = xkcd)
        
        errors = [ans] + errors
        
        
        distractors = [html_image(er, preview = (a_type == 'preview'), 
                                  width = 300, height = 300) 
                       for er in errors]
                       
        if a_type == 'preview':
            
            data = [[dist] for dist in distractors]            
            
            tb = Table(data, row_headers = ["Answer"] + ["Distractor"] * 4)            
            
            return question_stem + explanation + Table.get_style() + \
                tb.html()
            

        elif a_type == 'MC':
            if fmt == 'latex':
                question_stem = question_stem.replace("<div ","<p ")\
                        .replace("</div>","</p>")
                explanation = explanation.replace("<div ","<p ")\
                        .replace("</div>","</p>")
                distractors = [err.replace("<div ","<p ")\
                        .replace("</div>","</p>") for err in errors]
           
            
            question_stem = ' '.join(question_stem.split())
            distractors = [' '.join(err.split()) for err in errors]
            explanation = ' '.join(explanation.split()) + "\n"
            return tools.fully_formatted_question(question_stem, explanation, 
                                                  answer_choices=distractors)
            
      

if __name__ == "__main__":
    
    a_type = 'MC'
    fmt = 'latex'
    xkcd = False
    force = True
    
    prob = HyperbolaGraphProblem(seed=42) 
    
    pb = ""
    for i in range(10):
        if a_type == 'preview':
            pb += '<div class = \'posts\'>'
        pb += prob.stem(a_type=a_type, fmt = fmt, xkcd=xkcd)
        if a_type == 'preview':
            pb += '</div><br>'
    for i in range(5):
        if a_type  == 'preview':
            pb += '<div class = \'posts\'>'
        pb += prob.stem(a_type=a_type, fmt = fmt, expanded = False, xkcd=xkcd)
        if a_type == 'preview':
            pb += '</div><br>'
    print(pb)