# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:07:15 2016

@author: ketchers
"""


from __future__  import division
from __future__ import print_function
import numpy as np
import sys
import tools
import os
import os.path
import sympy as sym
import scipy as sp
import mpmath as mp
#import scipy.integrate as spi
import random
from html_tools import html_image
from table import Table
import pylab as plt
from R_tools import make_func


from ProblemTypes import *
from FunctionTypes import *

class ArcLengthGenerator(object):
    
    Q_TYPES = ['exact', 'numeric', 'integral']
    A_TYPES = ['MC', 'FR'] # 'MATCH' is proposed for the future
     
    # For grading purposes the "integral" choice will only be used with "MC"
    
    # So far I just have the ArcLengthProblemType1
    
    FUNCTYPES = [FunctionType1, FunctionType2, FunctionType3]
    
    def __init__(self, seed = None, path = 'arclength'):
        
        self.cache = set()
        
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            
        self.path = path
       
        
    def stem(self, a_type = None, q_type = None, 
             start = None, stop = None, ftype = None,
             preview = False, explanation_type = 'simple',
             **kwargs):

        """
        Named Parameters:
        ----------------
        
        a_type: str
            ["MC", "FR", "MATCH"]
        q_type: str 
            ['exact', 'numeric', 'integral']
        start, stop: numeric
            The endpoints on which the arc is defined.
        kwargs: 
            These are passed in if needed for the specific type of function.
        """
       
        kw_args = {
            'a_type': a_type,
            'q_type': q_type,
            'ftype':ftype,
            'start':start,
            'stop':stop,
            'preview':preview,
            'explanation_type':explanation_type
        }
        kw_args.update(kwargs)
    
    
        if a_type is None:
            a_type = random.choice(['MC']*3 + ['FR']) # about in 'FR' per three 'MC'    
            
        
        if ftype is None:
            ftype = random.choice(self.__class__.FUNCTYPES)
        
        if a_type == 'MC':
            if q_type is None:
                q_type = random.choice(ftype.IMPLEMENTS)
        
        if a_type == 'FR':
            if q_type is None:
                q_type = random.choice([tp for tp in ftype.IMPLEMENTS \
                    if tp != 'integral'])
          
        prob = ftype(**kw_args)
                
        if start is None:
            start, stop = random.choice(prob.BOUNDS)
    
    
        # This is for keeping track of what has been created
        pstring = str(prob) + " start = {start}, stop = {stop}, \
            a_type = {a_type}, q_type = {q_type}".format(start = start, 
                                                         stop = stop, 
                                                         a_type = a_type, 
                                                         q_type = q_type)
       
        phash = hash(pstring)
        if phash in self.cache:
            return self.stem(**kw_args) # try again
        
        question = """
            Compute the arc length of the curve C given by:
            <center><p width = 75%>
                {curve}
            </p></center>
            """.format(curve = prob.arc_desc(start, stop))
        
            
        
      
        
        if a_type == "MC":
            
            errors = ["$_"+sym.latex(err)+"$_" for err in \
                      prob.arc_length_errors(start, stop, q_type)]
                      
            explanation =  prob.arc_length_explain(start = start, stop = stop, 
                                           a_type = a_type)
            
            if q_type == 'exact':
                question += """
                    Choose the correct expression for the exact arc length.
                    """
            elif q_type == 'numeric':
                question += """
                    Choose the best numeric approximation to the arc length.
                    """
            else: #q_type == 'integral'
                question += """
                    Choose the integral that represents the arc length.
                    """

            question = ' '.join(question.split())
            distractors = [' '.join(err.split()) for err in errors]
            explanation = ' '.join(explanation.split()) + "\n"
            
            problem= tools.fully_formatted_question(question, 
                   explanation, 
                   answer_choices = errors)
                
        elif a_type == "FR":
            
            explanation = prob.arc_length_explain(start, stop, a_type = a_type)            
            
            if q_type == 'numeric':
                question += "Give the value of the arc length as a decimal\
                    number correct to at least two decimal places."
                         
                answer_mathml = tools.itex2mml("$_" +\
                    "%.3f"%(prob.arc_length(start, stop,'numeric')) + "$_")
                            
                
            elif q_type == 'exact':
                question += "Evaluate the arc length and provide the exact \
                    value."
                answer_mathml = tools.itex2mml("$_" + \
                    sym.latex(prob.arc_length(start, stop,'exact')) + "$_")
                    
            question = ' '.join(question.split())
            explanation = ' '.join(explanation.split()) + "\n"
                
            problem = tools.fully_formatted_question(question, 
                            explanation, 
                            answer_mathml)
      
        if preview:
            
            if a_type == 'MC':
                errors = prob.arc_length_errors(start, stop, q_type)
                errors = ''.join(['&nbsp;(%d)&nbsp; $_'%i + err + '$_ ' for i, err \
                                   in enumerate(errors)])
            else:
                errors = "Free response."                
                
            ret = """
                <style>
                div.foo {{
                    background-color: rgba(100,200,100,.5);
                    margin: 20px;
                    padding: 10px;
                    border-radius: 10px;
                }}     
                </style>
                <br><hr><br>
                <div class='foo'>
                <strong>Question: </strong>{question}
                <br><br>
                <strong>Answers: </strong>{errors}
                <br><br>
                <strong>Explanation:</strong><br>
                {explanation}        
                </div>
                """.format(explanation=prob.arc_length_explain(start,stop),
                           question = question,
                           errors = errors)
            return ret
            
        return problem

if __name__ == "__main__":
    gen =  ArcLengthGenerator()
    for i in range(40):
        a_type = random.choice(["MC"]*3+["FR"])
        ind_var = random.choice(['x']*3 + ['y'])
        dep_var = 'y' if ind_var == 'x' else 'y'
        print(gen.stem(a_type = a_type, 
                       ind_var=ind_var, dep_var=dep_var, 
                       preview = True))
        