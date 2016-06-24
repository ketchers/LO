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
    
    def stem(self, preview = False, xkcd = False, force = False):
        
        kwargs = {
            'preview':False
        }
        
        prob = Hyperbola()
        
        if prob in self.done:
            self.stem(**kwargs)
        else:
            self.done.add(prob)
            self.count += 1
        
        
        question_stem = 'Choose the graph of $$%s.$$' % (prob.latex)
        
        ans = prob.show(path=self.path, xkcd = xkcd, force = force)
                                                         
        explanation = prob.explanation(path=self.path + "/explanation", 
                                   preview = preview, xkcd = xkcd)
                                                         
        errors = self.gen_errors(prob, force = force, xkcd = xkcd)
        
        errors = [ans] + errors
        
        
        distractors = [html_image(er, preview = preview, width = 300, height = 300) 
                       for er in errors]               
        
        return tools.fully_formatted_question(question_stem, explanation, 
                                              answer_choices=distractors)

if __name__ == "__main__":
    
    preview = True
    xkcd = False
    force = True
    
    prob = HyperbolaGraphProblem(seed=42) 
    
    pb = ""
    for i in range(10):
        pb += '<p class = \'posts\'>'
        pb += prob.stem(preview=preview, xkcd=xkcd)
        pb += '</p><br><br><p class = \'posts\'>'
    
    print(pb)