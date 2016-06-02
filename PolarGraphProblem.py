
# coding: utf-8

# ## This is still in progress ....
# 

# ## We should be able to take care of all of these with one class.
# * 11.4.3a	Graph a circle or a line from a polar equation
# * 11.4.3b	Graph a cardioid from a polar equation
# * 11.4.3c	Graph a limacon from a polar equation
# * 11.4.3d	Graph a lemniscate from a polar equation
# 
# 
# # Some defs:
# * **circles:** These are graphs of $r = a \cos(\theta)$ or $r = a \sin(\theta)$
# * **Limacons:** These are all graphs of $r = a \pm b \cos(\theta)$ or 
#     $r = a \pm b \sin(\theta)$ where $a,b > 0$ is assumed. Note: $a = 0$ is the circle 
#     case.  
#     * **Cardioids:** These are determined by $a/b = 1$.
#     * **Limacons for one loop:** These are determined by $a/b > 1$.
#         * **Convex:** These are those with $a/b \ge 2$.
#     * **Limacons for two loops:** These are those with $a/b < 1$.
# * **Roses:** These are of the form $r = a \cos(n\cdot \theta)$ or $r = a \sin(n\cdot \theta)$
# * **Lemniscate:** These are of the form $r^2 = a^2\cos(2\theta)$ or $r^2=a^2\sin(2\theta)$

from __future__ import print_function
from __future__ import division
import sys
import os

import random
import sympy as sym
import numpy as np
import pylab as plt
import datetime

import tools
from PolarFunction import PolarFunction

class PolarGraphProblem():
    
    # Bad OOP style
    Q_TYPES = PolarFunction.TYPES
    
    # Dictionary of types 
    qtd = dict(enumerate(Q_TYPES))
    invqtd = dict([(qtd[i],i) for i in qtd])
 
    A_TYPES = ['MC', 'FR', 'Match'] # Match ... maybe later.
    
    def __init__(self, seed = None, path = 'polar_plots'):
        """
        As usual done is a list that keeps track of what problems have been generated.
        """
        self.done = set()
        self.count = 0
        
        self.path = path
        
        if seed is None:
            self.seed = datetime.datetime.now().microsecond
        else:
            self.seed = seed

        random.seed(seed)
    
    @staticmethod
    def get_types():
        [print(str(i) + ":" + str(PolarGraphProblem.qtd[i])) 
                for i in PolarGraphProblem.qtd]
    
    def gen_prob(self, q_type):
        # Generate our current problem
        prob = PolarFunction(f_type = q_type)
                
    def gen_errors(self, prob, error_types, force = False, rad = False,
                   include_image = False, xkcd = False):
        """
        This will generate the errors for MC.
        
        Parameters:
        ----------
            prob         : (q_type, a, b, trig) (the correct problem)
            error_types  : Besides instances of the same q_type with the same 
                           a, b generate some other reasonable options.
            force        : Regenerate images (passed down)
            include_image: (Boolean) For preview.
        """
        # Initially consider the variant of the original
        
        # Swap sin <-> cos
        if prob.f == 'sin':
            new_func = 'cos'
        elif prob.f == 'cos':
            new_func = 'sin'
        elif prob.f == 'sec':
            new_func = 'csc'
        elif prob.f == 'sec':
            new_func = 'csc'
        else:
            new_func = prob.f


        # Generate some distractors very similar to given        
        errors = [(-prob.a, prob.b, prob.n, prob.f, prob.f_type),
                  (prob.a, -prob.b, prob.n, prob.f, prob.f_type),
                  (prob.a, prob.b, prob.n, new_func, prob.f_type),
                  (-prob.a, prob.b, prob.n, new_func, prob.f_type),
                  (prob.a, -prob.b, prob.n, new_func, prob.f_type)]
        if prob.f_type == 'rose' and prob.n % 2 == 0 and prob.n > 2:
            errors += [(prob.a, prob.b, prob.n / 2, prob.f, prob.f_type)]
        elif prob.f_type == 'rose' and prob.n % 2 == 1:
            prob.errors += [(prob.a, prob.b, 2 * prob.n, prob.f, prob.f_type)]
        else:
            pass
        
                   
        for e in error_types:
            for k in range(4):
                f = PolarFunction(f_type = e)
                errors += [(f.a, f.b, f.n, f.f, f.f_type)]
        
        random.shuffle(errors)
        
        ers = []
        while(len(ers) < 4):
            er = errors.pop()
            f = PolarFunction(a = er[0], b = er[1], n = er[2], 
                            f = er[3], f_type = er[4])
            if len(ers) == 0 and f != prob:
                ers.append(f)
            elif reduce(lambda x, y: x & y, [f != g for g in ers], True)\
                and f != prob:
                ers.append(f)
            else:
                print("A function with same graph is present")
     
        # Change number of distractors here
        errors = [f.show(path = self.path, force = force, rad = rad) for f in ers]
        
        return errors
        
    
    
    def stem(self, q_type = None, a_type = 'MC', error_types = [], 
             force = False, rad = False, include_image = False,
             xkcd = False):
        """
        Parameters:
        ----------
            q_type        : This is the kind of polar graph
            a_type        : "MC" (multiple choice) for now. Matching coming soon?
            error_types   : When generating distractors, what other types of plots 
                            should show up?
            force         : (Boolean) Force regeneration of images.
            include_image : (Boolean) For previewing. 
        """
    
        kwargs = {
            'q_type': q_type,
            'a_type': a_type,
            'error_types': error_types,
            'force':force,
            'include_image':include_image,
            'xkcd':xkcd
        }
    
          
        
        prob = PolarFunction(f_type = q_type)
        
        if prob in self.done:
            self.stem(**kwargs)
        else:
            self.done.add(prob)
            self.count += 1
        
        
        question_stem = 'Choose the graph of $$%s.$$' % (prob.latex())
        
        ans = prob.show(path = self.path, rad = rad)
        
        explanation = prob.explain(path = self.path + "/explanation", 
                        include_image = include_image, rad = rad,
                        xkcd = xkcd)
        
        
        if a_type == 'MC':
            errors = self.gen_errors(prob, error_types, force = force, 
                            include_image = include_image, 
                            xkcd = xkcd, rad = rad)
            errors = [ans] + errors 
            if include_image:
                distractors = ["<img width = 25%s src=\'%s\'>" % ('%', er) for er in errors]               
            else:
                distractors = ["${%s}$" % (er) for er in errors]
            return tools.fully_formatted_question(question_stem, explanation, answer_choices=distractors)
        elif a_type == 'Match':
            pass
        else:
            pass
        
        
if __name__ == "__main__":
    pb = ""
    
    # I have xkcd turned on for fun, the default is off, but if you turn
    # it off you need to add force = True to force regeneration of images
    # This should run very fast after the first run.    
    
    preview = True
    xkcd = True
    
    for i in range(10):
        pb += '<p class = \'posts\'>'
        pb += PolarGraphProblem(seed = 42).stem(q_type = 'lemniscate',
                error_types = ['rose','circle'], 
                rad = random.choice([True, False]),
                include_image = preview, xkcd = xkcd)
        pb += '</p><br><br><p class = \'posts\'>'
        pb += PolarGraphProblem(seed = 43).stem(q_type = 'line',
                error_types = ['circle','line through the origin'],
                rad = random.choice([True, False]),
                include_image = preview, xkcd = xkcd)
        pb += '</p><br><br><p class = \'posts\'>'        
        pb += PolarGraphProblem(seed = 44).stem(q_type = '(convex one-loop) limacon',
                error_types = ['circle','(dimpled one-loop) limacon'],
                rad = random.choice([True, False]),
                include_image = preview, xkcd = xkcd)
        pb += '</p><br><br>'
    
    # The following formats things for preview.
    print(pb.replace('\n',""))
 