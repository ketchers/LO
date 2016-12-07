# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:56:20 2016

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
from ProblemTypes import ArcLengthProb

#%%
class FunctionType1(object):
    """
    This is an exact form:
        a * 1/(p+1) * u^{p+1} - 1/(4a) * 1/(1 - p) * u^{1-p} for p != 1
        a/2 * u^2 - 1/(4a) * ln(u) for p = 1  
        
    Usage: f = FunctionTypeType1(a = "2/3", p = "1/2", start = 1, stop = 2)
    -----
    
    f = FunctionType1("2/3", "1/2", ind_var = 'u', dep_var = 'r', start = 1, stop = x)
    f = FunctionType1(a = "1/3", p = 1)
    f = FunctionType() # Random problem

    Parameters:
    ----------
    (kw) a, p: str
        These should be like a = "2/3", p = "1/2".
    (kw) ind_var, dep_var: str 
        For example ind_var = 'x', dep_var = 'y'. This is the independent/dependent variable 
        for the problem. 
        
        
    If some of these are unset, reasonable random assignments will be made.
    """
    
    IMPLEMENTS = ['exact', 'integral', 'numeric']
    
    def __init__(self, a = None, p = None, ind_var = None, 
                 dep_var = None, **kwargs):
        
        self.__dict__.update(kwargs)
        
        self.a_type = kwargs.get('a_type', 'preview')        
        self.q_type = kwargs.get('q_type', None)  
        
        # If dep_var/ind_var are not specified fix some defaults
        my_vars = ['x','u','v','y','w']
        random.shuffle(my_vars)

        if (ind_var, dep_var) == (None, None):
            ind_var = my_vars.pop()
            dep_var = my_vars.pop()
        elif ind_var is None and dep_var is not None:
            my_vars.remove(dep_var)
            ind_var = my_vars.pop()
        elif dep_var is None and ind_var is not None:
            my_vars.remove(ind_var)
            dep_var = my_vars.pop()    
        
        self.ind_var = ind_var
        self.dep_var = dep_var
    
        
        self.u, self.v_ = sym.symbols(ind_var + " " + dep_var)
        u = self.u
        
        # set defaults for a and p if none are provided
        if p is None:
            p = random.choice(["1","1","1/2","1/3","2/3","1/4","3/4","2"])
        if a is None:
            a = random.choice(["1/2","-1/2","2/3","-2/3","1/4",
                               "-1/4","2","-2","3","-3"])
        
        self.p = sym.sympify(p)
        self.a = sym.sympify(a)
        a = self.a
        p = self.p
        
        self.BOUNDS = [(l, l + h) for l in range(1, 5) for h in range(2, 5)]
        
        self.kwargs = kwargs
        self.kwargs.update({
            'a': a,
            'p': p,
            'ind_var':ind_var,
            'dep_var':dep_var,
        })
        
        
        if p != 1:
            self.v = a * 1 / (p + 1) * u**(p + 1) - 1 / (4 * a) * 1 / (1 - p) * u**(1 - p)
            self.Dv = sym.Derivative(self.v)
            self.dv = a * u**p - 1/(4 * a) * u**(-p)
            self.ds = sym.Abs(a) * u**p + 1/(4 * sym.Abs(a)) * u**(-p)
            self.Ids = sym.Abs(a)/(p+1) * u **(p+1) + 1/(4*sym.Abs(a)*(1-p))*u**(1-p)
           
        else:
            self.v = a/2 * u**2 - 1 / (4 * a) * sym.ln(u) 
            self.Dv = sym.Derivative(self.v)
            self.dv = a * u - 1 / (4 * a * u)
            self.ds = sym.Abs(a) * u + 1 / (4 * sym.Abs(a) * u) 
            self.Ids = sym.Abs(a)/2 * u**2 + 1/(4 * sym.Abs(a)) * sym.ln(u)
       

    def __call__(self, inp):
        f = make_func(str(self.v), func_params=(self.ind_var), func_type='numpy')
        return f(inp)
        
    def __str__(self):
        return str(self.v)
        
    # Note: f = FunctionType1()
    #       f == eval(repr(f)) -> true!
    def __repr__(self):
        hstr = str(self.__class__.__name__) + "(a = \"" + str(self.a) + \
            "\", p = \"" + str(self.p) + "\", ind_var = \"" + str(self.ind_var) + \
            "\", dep_var = \"" + str(self.dep_var) + "\")"
        return hstr

    def __hash__(self):

        if self.hash == 0:
            self.hash = hash(self.__repr__())
        return self.hash

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return false
        else:
            return self.a == other.a and self.p == other.p and \
                    self.ind_var == other.ind_var and self.dep_var == other.dep_var

    def __ne__(self, other):
        return not self.__eq__(other)
        
    # Stuff specific to arc length
        
    def arc_length(self, start, stop, q_type = None):
        """
        Parameters:
        ---------
        q_type:  string
            'exact', 'integral', 'numeric'
        start, stop: numeric/str
            These are the limits of integration
        """
        u = self.u
        
        if q_type is None:
            q_type = self.q_type
        
        if q_type == 'integral':
            return sym.Integral(self.ds, (u, start, stop))
        elif q_type == 'exact':
            return self.Ids.subs(u, stop) - self.Ids.subs(u, start)
        else:
            f = make_func(str(self.ds), func_params=(self.ind_var), func_type='numpy')
            value = mp.quad(f, [start, stop])
            return value
        
    def arc_desc(self, start, stop):
        desc = """
            The portion of the curve $_{dep_var} = {eqn}$_ 
            where $_{start} \\leq {ind_var} \\leq {stop}$_.
            """.format(dep_var = self.dep_var, 
                    eqn = sym.latex(self.v), 
                    start = start, 
                    ind_var = self.ind_var, 
                    stop = stop)
        return desc
        
    def arc_length_errors(self, start, stop, q_type = None):
        """
        e_type: str
            Any of 'exact', 'integral', 'numeric'
        """
        a = self.a
        b = self.p
        v = self.v
        u = self.u
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
        if q_type is None:
            q_type = self.q_type
        
       
        errors = []
        
        if q_type == 'exact':
      
            # Forget to add 1
            errors.append(sym.latex(self.v.subs(u, stop) - 
                                    self.v.subs(u, start))\
                          .replace("log","ln"))
            # Forget to take sqrt
            errors.append(sym.latex(sym.Abs((1 + (self.dv**2).expand()).subs(u, stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start)))\
                          .replace("log","ln"))
            # Take sqrt after evaluating
            errors.append(sym.latex(sym.Abs(sym.sqrt((1 + (self.dv**2).expand()).subs(u,stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start))))\
                          .replace("log","ln"))
            # Add instead of subtract end points
            errors.append(sym.latex(sym.Abs(self.Ids.subs(u, stop) + 
                                    self.Ids.subs(u, start)))\
                          .replace("\\log", "\\ln"))
            # Just be off by some random amount
            errs = ["1/2", "-1/2", "1/3", "-1/3", "1/4", "-1/4"]
            a_new = a * (1 + sym.sympify(random.choice(errs)))
            b_new = b * (1 + sym.sympify(random.choice(errs)))
           
                
            kwargs_new = {}
            kwargs_new.update(self.kwargs)
            kwargs_new.update({'a':a_new, 'p':b_new})
            # Build new copy of self with different parameters.
            err = self.__class__(**kwargs_new).arc_length(start, stop,'exact')
            errors.append(sym.latex(err).replace("\\log", "\\ln"))
            ans = sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln")
          
        if q_type == 'integral':
            
            start_ = sym.latex(start)
            stop_ = sym.latex(stop)
            
            c = sym.sympify("1/2")
            
            # Forget to add 1
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex((self.dv**2).expand()), u = u))
            # Forget to square dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + self.dv), u = u))
            # Forget to take sqrt
            errors.append("\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Take the square root after integrating
            errors.append("\\left(\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}\\right)^{{1/2}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Use v instead of dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + v**2), u = u))
            ans = sym.latex(sym.Integral((1 + (self.ds**2).expand())**c, (u, start, stop)))
           
        
        if q_type == 'numeric':
            
            start_ = start.evalf()
            stop_ = stop.evalf()
            
            # Forget to add 1
            f = make_func(str(sym.Abs(self.dv)), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to square dv/du
            f = make_func(str(sym.Abs((1 + self.dv)**0.5)), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to take sqrt
            f = make_func(str(1 + self.dv**2), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Just be off by some amount
            value = self.arc_length(start_, stop_,'numeric')
            vals = [value * (1.1), value*(0.9), value + 0.1, value - 0.1]
            vals = [sym.Abs(foo) for foo in vals if foo > 0]
            errors.append('%.3f'%(random.choice(vals)))
            # Take the sqrt after integrating
            f = make_func(str(1 + self.dv**2), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%((mp.quad(f, [start_, stop_]))**(1/2)))
            
            
            ans = "%.3f"%(self.arc_length(start, stop,'numeric'))
        
        
        errors = set(errors) - {ans}
        errors = list(errors)
        random.shuffle(errors)
        
        errors = [ans] + errors[:3] #Change 3 to 4, 5 or 6 for more
        return errors
        
        

    def arc_length_explain(self, start, stop, a_type = None,
                           explanation_type=None, preview = None):
        
        
        if a_type is None:
            a_type = self.a_type
            
        v = self.v_
        u = self.u
        
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
       
        if explanation_type is None:
            explanation_type = self.kwargs.get('explanation_type', 'simple')
        
        if preview is None:
            preview = self.kwargs.get('preview', False)  
       
        
        explanation = ArcLengthProb(path='arc_length/explanation', 
                    a_type = a_type, 
                    explanation_type = explanation_type,
                    preview = preview).explain()
                    
        explanation += """
            <p>
            For this problem the curve is $_C$_ is given by $_%s$_
            with the independent variable being $_%s$_ on the interval 
            $_[%s,%s]$_. So the arc length is
            $$s =\\int_{%s}^{%s} \sqrt{1+\\left(\\frac{d%s}{d%s}\\right)^2}\,d%s$$
            $$\\frac{d%s}{d%s} = %s = %s$$
            \\begin{align}
              \\sqrt{1 + \\left(%s\\right)^2} &= \\sqrt{%s} \\\\
                &= \sqrt{\\left(%s\\right)^2} = %s
            \\end{align}
            </p>
        """%tuple(map(lambda x: sym.latex(x).replace("\\log","\\ln"), 
                      [sym.Eq(v, self.v), u, start, stop, start, stop, 
                       v, u, u, v, u, 
                       self.Dv, self.dv, self.dv, 1 + (self.dv**2).expand(), 
                        self.ds, self.ds]))
        
        aa = [sym.latex(self.arc_length(start, stop,'integral')), 
              "\\left." + sym.latex(self.Ids).replace('\\log', '\\ln') +
              "\\right|_{%s=%s}^{%s=%s}"%(u, start, u, stop),
             sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln") + 
             '\\approx %.3f'%(self.arc_length(start, stop,'numeric'))]
        
        # Add self.numeric / self.anti 
        
        explanation += """
            <p>
            Thus the arclength is given by
            %s
            </p>
        """%(tools.align(*aa))
        #
        return explanation
#%%


#%%
class FunctionType2(object):
    """
    This is an exact form:
        a + ln(b*sin(t)), a + ln(b*cos(t)), a + ln(b*sec(t)), a + ln(b*csc(t))
        
    Usage: f = FunctionTypeType1(a = "2/3", b = "1/2", ind_var = 't', dep_var = 'r')
    -----
    
    f  = FunctionType1("2/3", "1/2", ind_var = 'u', dep_var = 'r')
    f = FunctionType1() # Random

    Parameters:
    ----------
    (kw) a, b: str
        These should be like a = "2/3", b = "1/2".
    (kw) ind_var, dep_var: str 
        For example ind_var = 'x', dep_var = 'y'. This is the independent/dependent variable 
        for the problem. 
        
        
    If some of these are unset, reasonable random assignments will be made.
    """
    
    IMPLEMENTS = ['exact', 'integral', 'numeric']
    
    def __init__(self, a = None, b = None, ind_var = None, 
                 dep_var = None, trig = None, **kwargs):
                     
        self.__dict__.update(kwargs)
                     
        self.a_type = kwargs.get('a_type', 'preview')        
        self.q_type = kwargs.get('q_type', None)  
              
        
        # If dep_var/ind_var are not specified fix some defaults
        my_vars = ['x','u','v','y','w']
        random.shuffle(my_vars)

        if (ind_var, dep_var) == (None, None):
            ind_var = my_vars.pop()
            dep_var = my_vars.pop()
        elif ind_var is None and dep_var is not None:
            my_vars.remove(dep_var)
            ind_var = my_vars.pop()
        elif dep_var is None and ind_var is not None:
            my_vars.remove(ind_var)
            dep_var = my_vars.pop()    
        
        self.ind_var = ind_var
        self.dep_var = dep_var
    
        
        self.u, self.v_ = sym.symbols(ind_var + " " + dep_var)
        u = self.u
        
        # set defaults for a and p if none are provided
        if b is None:
            b = random.choice(["1","2", "3"])
        if a is None:
            a = random.choice(["1", "2", "3", "4"])
        
        if trig is None:
            trig = random.choice(['sin', 'cos', 'sec', 'csc'])
            
        self.trig = trig
        
        self.b = sym.sympify(b)
        self.a = sym.sympify(a)
        a = self.a
        b = self.b
        
        
        self.BOUNDS = [(sym.sympify(l), sym.sympify(h)) for l, h in 
                       [("pi/6", "pi/2"), ("pi/6", "5*pi/6"), ("pi/6", "2*pi/3"), 
                        ("pi/4", "pi/2"), ("pi/4", "2*pi/3"), ("pi/3", "pi/2"),
                        ("pi/3", "2*pi/3"), ("pi/2", "5*pi/6")]]
        
        if trig in ['sec', 'cos']:
            self.BOUNDS = [(l - sym.sympify("pi/2"), h - sym.sympify("pi/2")) \
                           for l, h in self.BOUNDS]
            
        
        self.kwargs = {
            'a': a,
            'b': b,
            'ind_var':ind_var,
            'dep_var':dep_var,
            'trig':trig
        }
        
        trig_ = sym.sympify(trig + "(" + str(u) + ")")
        self.v = a + sym.log(b * trig_)
        self.Dv = sym.Derivative(self.v)
        
        if trig in ['sec', 'cos']:
            self.dv = sym.tan(u) * (-1 if trig == 'cos' else 1)
            self.ds = sym.sec(u)
            self.Ids = sym.log(sym.Abs(sym.sec(u) + sym.tan(u)))
           
        else:
            self.dv = sym.cot(u) * (-1 if trig == 'cot' else 1)
            self.ds = sym.csc(u)
            self.Ids = -sym.log(sym.Abs(sym.csc(u) + sym.cot(u)))
       

    def __call__(self, inp):
        f = make_func(str(self.v), func_params=(self.ind_var), func_type='numpy')
        return f(inp)
        
    def __str__(self):
        return str(self.v)
        
    # Note: f = FunctionType1()
    #       f == eval(repr(f)) -> true!
    def __repr__(self):
        hstr = str(self.__class__.__name__) + "(a = \"" + str(self.a) + \
            "\", b = \"" + str(self.b) + "\", ind_var = \"" + str(self.ind_var) + \
            "\", dep_var = \"" + str(self.dep_var) + "\", trig = \"" + self.trig + "\")"
        return hstr

    def __hash__(self):

        if self.hash == 0:
            self.hash = hash(self.__repr__())
        return self.hash

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return false
        else:
            return self.a == other.a and self.b == other.b and \
                    self.ind_var == other.ind_var and self.dep_var == other.dep_var\
                    and self.trig == other.trig

    def __ne__(self, other):
        return not self.__eq__(other)
        
    # Stuff specific to arc length
        
    def arc_length(self, start, stop, q_type = None):
        """
        Parameters:
        ---------
        q_type:  string
            'exact', 'integral', 'numeric'
        start, stop: numeric/str
            These are the limits of integration
        """
        u = self.u
        
        if q_type is None:
            q_type = self.q_type
        
        start = sym.sympify(start)
        stop = sym.sympify(stop)  
        
        if q_type == 'integral':
            return sym.Integral(self.ds, (u, start, stop))
        elif q_type == 'exact':
            return self.Ids.subs(u, stop) - self.Ids.subs(u, start)
        else:
            f = make_func(str(self.ds), func_params=(self.ind_var), func_type='numpy')
            value = mp.quad(f, [start.evalf(), stop.evalf()])
            return value
        
    def arc_desc(self, start, stop):
        desc = """
            The portion of the curve $_{dep_var} = {eqn}$_ 
            where $_{start} \\leq {ind_var} \\leq {stop}$_.
            """.format(dep_var = self.dep_var, 
                    eqn = sym.latex(self.v), 
                    start = sym.latex(start), 
                    ind_var = self.ind_var, 
                    stop = sym.latex(stop))
        return desc
        
    def arc_length_errors(self, start, stop, q_type = None):
        """
        e_type: str
            Any of 'exact', 'integral', 'numeric'
        """
        a = self.a
        b = self.b
        v = self.v
        u = self.u
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
        
        
        if q_type is None:
            q_type = self.q_type
        
        
            
       
        
        errors = []
        if q_type == 'exact':
      
            # Forget to add 1
            errors.append(sym.latex(self.v.subs(u, stop) - 
                                    self.v.subs(u, start))\
                          .replace("log","ln"))
            # Forget to take sqrt
            errors.append(sym.latex(sym.Abs((1 + (self.dv**2).expand()).subs(u, stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start)))\
                          .replace("log","ln"))
            # Take sqrt after evaluating
            errors.append(sym.latex(sym.Abs(sym.sqrt((1 + (self.dv**2).expand()).subs(u,stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start))))\
                          .replace("log","ln"))
            # Add instead of subtract end points
            errors.append(sym.latex(sym.Abs(self.Ids.subs(u, stop) + 
                                    self.Ids.subs(u, start)))\
                          .replace("\\log", "\\ln"))
            # Just be off by some random amount
            errs = ["1/2", "-1/2", "1/3", "-1/3", "1/4", "-1/4"]
            a_new = a * (1 + sym.sympify(random.choice(errs)))
            b_new = b * (1 + sym.sympify(random.choice(errs)))
           
                
            kwargs_new = {}
            kwargs_new.update(self.kwargs)
            kwargs_new.update({'a':a_new, 'b':b_new})
            # Build new copy of self with different parameters.
            err = self.__class__(**kwargs_new).arc_length(start, stop,'exact')
            errors.append(sym.latex(err).replace("\\log", "\\ln"))
            ans = sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln")
          
        if q_type == 'integral':
            
            start_ = sym.latex(start)
            stop_ = sym.latex(stop)
            
            c = sym.sympify("1/2")
            
            # Forget to add 1
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex((self.dv**2).expand()), u = u))
            # Forget to square dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + self.dv), u = u))
            # Forget to take sqrt
            errors.append("\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Take the square root after integrating
            errors.append("\\left(\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}\\right)^{{1/2}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Use v instead of dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + v**2), u = u))
            ans = sym.latex(sym.Integral((1 + (self.ds**2).expand())**c, (u, start, stop)))
           
        
        if q_type == 'numeric':
            
            start_ = start.evalf()
            stop_ = stop.evalf()
            
            # Forget to add 1
            func_desc = str(sym.Abs(self.dv))
            f = make_func(func_desc, func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to square dv/du
            func_desc = str(sym.Abs((1 + self.dv)**0.5))
            f = make_func(func_desc, func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to take sqrt
            func_desc = str(1 + self.dv**2)
            f = make_func(func_desc, func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Just be off by some amount
            value = self.arc_length(start_, stop_,'numeric')
            vals = [value * (1.1), value*(0.9), value + 0.1, value - 0.1]
            vals = [sym.Abs(b) for b in vals if b > 0]
            errors.append('%.3f'%(random.choice(vals)))
            # Take the sqrt after integrating
            func_desc = str(1 + self.dv**2)
            f = make_func(func_desc, func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%((mp.quad(f, [start_, stop_]))**(1/2)))
            
            
            ans = "%.3f"%(self.arc_length(start, stop,'numeric'))
        
        
        errors = set(errors) - {ans}
        errors = list(errors)
        random.shuffle(errors)
        
        errors = [ans] + errors[:3] #Change 3 to 4, 5 or 6 for more
        return errors
        

    def arc_length_explain(self, start, stop, a_type = None,
                           explanation_type=None, preview = None):
        v = self.v_
        u = self.u
        
        if explanation_type is None:
            explanation_type = self.__dict__.get('explanation_type', 'simple')
        if preview is None:
            preview = self.__dict__.get('preview', False)         
        
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
        if a_type is None:
            a_type = self.a_type
        
        explanation = ArcLengthProb(path='arc_length/explanation', 
                    a_type = a_type, 
                    explanation_type = explanation_type,
                    preview = preview).explain()
                    
        explanation += """
            <p>
            For this problem the curve is $_C$_ is given by $_%s$_
            with the independent variable being $_%s$_ on the interval 
            $_[%s,%s]$_. So the arc length is
            $$s =\\int_{%s}^{%s} \sqrt{1+\\left(\\frac{d%s}{d%s}\\right)^2}\,d%s$$
            $$\\frac{d%s}{d%s} = %s = %s$$
            \\begin{align}
              \\sqrt{1 + \\left(%s\\right)^2} &= \\sqrt{%s} \\\\
                &= \sqrt{\\left(%s\\right)^2} = %s
            \\end{align}
            </p>
        """%tuple(map(lambda x: sym.latex(x).replace("\\log","\\ln"), 
                      [sym.Eq(v, self.v), u, start, stop, start, stop, 
                       v, u, u, v, u, 
                       self.Dv, self.dv, self.dv, 1 + (self.dv**2).expand(), 
                        self.ds, self.ds]))
        
        aa = [sym.latex(self.arc_length(start, stop, 'integral')), 
              "\\left." + sym.latex(self.Ids).replace('\\log', '\\ln') +
              "\\right|_{%s=%s}^{%s=%s}"%(u, sym.latex(start), u, sym.latex(stop)),
             sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln") + 
             '\\approx %.3f'%(self.arc_length(start, stop,'numeric'))]
        
        # Add self.numeric / self.anti 
        
        explanation += """
            <p>
            Thus the arclength is given by
            %s
            </p>
        """%(tools.align(*aa))
        #
        return explanation



#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:56:20 2016

@author: ketchers
"""


#%%
class FunctionType3(object):
    """
    This is an exact form:
        (ax + b)^(3/2)
        
    Usage: f = FunctionTypeType1(a = "2/3", b = "1/2", ind_var = 't', dep_var = 'r')
    -----
    
    f  = FunctionType1("2/3", "1/2", ind_var = 'u', dep_var = 'r')
    f = FunctionType1() # Random

    Parameters:
    ----------
    (kw) a, b: str
        These should be like a = "2/3", b = "1/2".
    (kw) ind_var, dep_var: str 
        For example ind_var = 'x', dep_var = 'y'. This is the independent/dependent variable 
        for the problem. 
        
        
    If some of these are unset, reasonable random assignments will be made.
    """
    
    IMPLEMENTS = ['exact', 'integral', 'numeric']
    
    def __init__(self, a = None, b = None, ind_var = None, 
                 dep_var = None, trig = None, **kwargs):
                     
        self.__dict__.update(kwargs)
                     
        self.a_type = kwargs.get('a_type', 'preview')        
        self.q_type = kwargs.get('q_type', None)  
        
        
        # If dep_var/ind_var are not specified fix some defaults
        my_vars = ['x','u','v','y','w']
        random.shuffle(my_vars)

        if (ind_var, dep_var) == (None, None):
            ind_var = my_vars.pop()
            dep_var = my_vars.pop()
        elif ind_var is None and dep_var is not None:
            my_vars.remove(dep_var)
            ind_var = my_vars.pop()
        elif dep_var is None and ind_var is not None:
            my_vars.remove(ind_var)
            dep_var = my_vars.pop()    
        
        self.ind_var = ind_var
        self.dep_var = dep_var
    
        
        self.u, self.v_, self.s  = sym.symbols(ind_var + " " + dep_var + " s" )
        u = self.u
        s = self.s
        
        # set defaults for a and p if none are provided
        if b is None:
            b = random.choice(["1","2", "1/2", "3", "-1/2", "-1", "-2", "-3"])
        if a is None:
            a = random.choice(["1", "2", "1/2", "1/3", "-1", "-2", "-1/2"])
        
        
        self.b = sym.sympify(b)
        self.a = sym.sympify(a)
        a = self.a
        b = self.b
        
        # Pick some safe limits of integration for random problems
        
        
        if a < 0:
            bd = min([int(-b/a - 4/(9*a**3)), -b/a])
            self.BOUNDS = [(l - h,l) for l in range(bd - 3, bd + 1) for h in range(1, 4)]
        else:
            bd = max([int(-b/a - 4/(9*a**3)), -b/a])
            self.BOUNDS = [(l,l + h) for l in range(bd + 1, bd + 4) for h in range(1, 4)]
        
        
            
        
        self.kwargs = {
            'a': a,
            'b': b,
            'ind_var':ind_var,
            'dep_var':dep_var,
            'trig':trig
        }
        
       
        self.v = (a*u + b)**(sym.sympify("3/2"))
        self.Dv = sym.Derivative(self.v)
        self.dv = self.Dv.doit()
        self.ds = sym.sqrt(1 + self.dv**2)
        self.Ids = sym.Integral(self.ds).doit()
       

    def __call__(self, inp):
        f = make_func(str(self.v), func_params=(self.ind_var), func_type='numpy')
        return f(inp)
        
    def __str__(self):
        return str(self.v)
        
    # Note: f = FunctionType1()
    #       f == eval(repr(f)) -> true!
    def __repr__(self):
        hstr = str(self.__class__.__name__) + "(a = \"" + str(self.a) + \
            "\", b = \"" + str(self.b) + "\", ind_var = \"" + str(self.ind_var) + \
            "\", dep_var = \"" + str(self.dep_var) + "\")"
        return hstr

    def __hash__(self):

        if self.hash == 0:
            self.hash = hash(self.__repr__())
        return self.hash

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return false
        else:
            return self.a == other.a and self.b == other.b and \
                    self.ind_var == other.ind_var and self.dep_var == other.dep_var
                 

    def __ne__(self, other):
        return not self.__eq__(other)
        
    # Stuff specific to arc length
        
    def arc_length(self, start, stop, q_type = None):
        """
        Parameters:
        ---------
        q_type:  string
            'exact', 'integral', 'numeric'
        start, stop: numeric/str
            These are the limits of integration
        """
        
        start = sym.sympify(start)
        stop = sym.sympify(stop)        
        
           
        
        if q_type is None:
            q_type = self.q_type        
        
        u = self.u
        I = sym.Integral(self.ds, (u, start, stop))
        
        if q_type == 'integral':
            return I
        elif q_type == 'exact':
            return I.doit()
        else:
            f = make_func(str(self.ds), func_params=(self.ind_var), func_type='numpy')
            value = mp.quad(f, [float(start.evalf()), float(stop.evalf())])
            return value
        
    def arc_desc(self, start, stop):
        desc = """
            The portion of the curve $_{dep_var} = {eqn}$_ 
            where $_{start} \\leq {ind_var} \\leq {stop}$_.
            """.format(dep_var = self.dep_var, 
                    eqn = sym.latex(self.v), 
                    start = start, 
                    ind_var = self.ind_var, 
                    stop = stop)
        return desc
        
    def arc_length_errors(self, start, stop, q_type = None):
        """
        e_type: str
            Any of 'exact', 'integral', 'numeric'
        """
        a = self.a
        b = self.b
        v = self.v
        u = self.u
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
        if q_type is None:
            q_type = self.q_type
        
         
        
        errors = []
        if q_type == 'exact':
      
            # Forget to add 1
            errors.append(sym.latex(self.v.subs(u, stop) - 
                                    self.v.subs(u, start))\
                          .replace("log","ln"))
            # Forget to take sqrt
            errors.append(sym.latex(sym.Abs((1 + (self.dv**2).expand()).subs(u, stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start)))\
                          .replace("log","ln"))
            # Take sqrt after evaluating
            errors.append(sym.latex(sym.Abs(sym.sqrt((1 + (self.dv**2).expand()).subs(u,stop) - 
                                    (1 + (self.dv**2).expand()).subs(u, start))))\
                          .replace("log","ln"))
            # Add instead of subtract end points
            errors.append(sym.latex(sym.Abs(self.Ids.subs(u, stop) + 
                                    self.Ids.subs(u, start)))\
                          .replace("\\log", "\\ln"))
            # Just be off by some random amount
            errs = ["1/2", "-1/2", "1/3", "-1/3", "1/4", "-1/4"]
            a_new = a * (1 + sym.sympify(random.choice(errs)))
            b_new = b * (1 + sym.sympify(random.choice(errs)))
           
                
            kwargs_new = {}
            kwargs_new.update(self.kwargs)
            kwargs_new.update({'a':a_new, 'b':b_new})
            # Build new copy of self with different parameters.
            err = self.__class__(**kwargs_new).arc_length(start, stop,'exact')
            errors.append(sym.latex(err).replace("\\log", "\\ln"))
            ans = sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln")
          
        if q_type == 'integral':
            
            start_ = sym.latex(start)
            stop_ = sym.latex(stop)
            
            c = sym.sympify("1/2")
            
            # Forget to add 1
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex((self.dv**2).expand()), u = u))
            # Forget to square dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + self.dv), u = u))
            # Forget to take sqrt
            errors.append("\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Take the square root after integrating
            errors.append("\\left(\\int_{{{start}}}^{{{end}}}{form}\\,d{{{u}}}\\right)^{{1/2}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + (self.dv**2).expand()), u = u))
            # Use v instead of dv/du
            errors.append("\\int_{{{start}}}^{{{end}}}\\sqrt{{{form}}}\\,d{{{u}}}"\
                          .format(start=start_,
                end=stop_, form=sym.latex(1 + v**2), u = u))
            ans = sym.latex(sym.Integral((1 + (self.ds**2).expand())**c, (u, start, stop)))
           
        
        if q_type == 'numeric':
            
            
            start_ = start.evalf()
            stop_ = stop.evalf()
            
            # Forget to add 1
            f = make_func(str(sym.Abs(self.dv)), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to square dv/du
            f = make_func(str(sym.Abs((1 + self.dv)**0.5)), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Forget to take sqrt
            f = make_func(str(1 + self.dv**2), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%(mp.quad(f, [start_, stop_])))
            # Just be off by some amount
            value = self.arc_length(start_, stop_,'numeric')
            vals = [value * (1.1), value*(0.9), value + 0.1, value - 0.1]
            vals = [sym.Abs(b) for b in vals if b > 0]
            errors.append('%.3f'%(random.choice(vals)))
            # Take the sqrt after integrating
            f = make_func(str(1 + self.dv**2), func_params=(self.ind_var), 
                          func_type='numpy')
            errors.append('%.3f'%((mp.quad(f, [start_, stop_]))**(1/2)))
            
            
            ans = "%.3f"%(self.arc_length(start, stop,'numeric'))
        
        
        errors = set(errors) - {ans}
        errors = list(errors)
        random.shuffle(errors)
        
        errors = [ans] + errors[:3] #Change 3 to 4, 5 or 6 for more
        return errors
        

    def arc_length_explain(self, start, stop, a_type = None,
                           explanation_type=None, preview = None):
        v = self.v_
        u = self.u
        
        if explanation_type is None:
            explanation_type = self.__dict__.get('explanation_type', 'simple')
        if preview is None:
            preview = self.__dict__.get('preview', False)
        
        start = sym.sympify(start)
        stop = sym.sympify(stop)
        
        if a_type is None:
            a_type = self.a_type
        
        explanation = ArcLengthProb(path='arc_length/explanation', 
                    a_type = a_type, 
                    explanation_type = explanation_type,
                    preview = preview).explain()
                    
        explanation += """
            <p>
            For this problem the curve is $_C$_ is given by $_%s$_
            with the independent variable being $_%s$_ on the interval 
            $_[%s,%s]$_. So the arc length is
            $$s =\\int_{%s}^{%s} \sqrt{1+\\left(\\frac{d%s}{d%s}\\right)^2}\,d%s$$
            $$\\frac{d%s}{d%s} = %s = %s$$
            \\begin{align}
              \\sqrt{1 + \\left(%s\\right)^2} &= \\sqrt{%s} \\\\
                &= \sqrt{\\left(%s\\right)^2} = %s
            \\end{align}
            </p>
        """%tuple(map(lambda x: sym.latex(x).replace("\\log","\\ln"), 
                      [sym.Eq(v, self.v), u, start, stop, start, stop, 
                       v, u, u, v, u, 
                       self.Dv, self.dv, self.dv, 1 + (self.dv**2).expand(), 
                        self.ds, self.ds]))
        
        ex = sym.Derivative(self.v).doit()**2 + 1
        eta = sym.symbols("eta")
        Isub = sym.Integral(sym.sqrt(ex),(u,start,stop)).transform(ex,eta)
        
        Isub_ = sym.Integral(sym.sqrt(ex)).transform(ex,eta).doit()
        
        
        aa = [sym.latex(self.arc_length(start, stop,'integral')),
              sym.latex(Isub) + "&\\text{ Substitute } \\eta = %s"%sym.latex(ex),
              "\\left." + sym.latex(Isub_) +
              "\\right|_{%s=%s}^{%s=%s}"%(u, sym.latex(ex.subs([(u,start)])), 
                                          u, sym.latex(ex.subs([(u, stop)]))),
             sym.latex(self.arc_length(start, stop,'exact')).replace("\\log", "\\ln") + 
             '\\approx %.3f'%(self.arc_length(start, stop,'numeric'))]
        
        # Add self.numeric / self.anti 
        
        explanation += """
            <p>
            Thus the arclength is given by
            %s
            </p>
        """%(tools.align(*aa))
        #
        return explanation

#%%
        
if __name__ == "__main__":
   
    
    f = eval("FunctionType3(a = \"-1\", b = \"1\", ind_var = \"x\", dep_var = \"y\")")
    f = FunctionType1(a = "-1", b = "1", ind_var = "x", dep_var = "y", trig = "csc")
    start, stop  = random.choice(f.BOUNDS)
    #print(start, stop)
    #print(f3("pi/3"))
    print(f.arc_length_explain(start, stop,'numeric', 
                                explanation_type = "comp", preview=False))
    #print(f3.arc_length(start, stop, 'numeric'))
    #print(f3.BOUNDS)

    #start, stop = random.choice(f2.BOUNDS)
    #print(f2.arc_length_explain(start, stop))
    #print(f2.arc_length_errors(4,8,'exact'))