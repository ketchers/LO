from __future__ import division
from __future__ import print_function
import sys
import os
import random 
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import tools
from R_tools import make_func
import re
from urllib import quote_plus


class PolarFunction():
    
    """
    The purpose of this class is to make dealing with polar functions easy.
    These function have the  form: 
         
         r = f(theta)
         
         or
         
         r^2 = f(theta)
    
    Precondition: Either f_name or _all_ of a,b,n,f should be given even if some are 0.
    
    Named Parameters:
    ----------------
        a      -- These should all be obvious
        b      --
        n      --
        f      -- 'sin', 'cos', 'sec', 'csc', 0 or 1 (for now). If f = 0, then 
                  we interpret this as 'theta = a' (this is a silly, but I'm not sure 
                  what a better fix is.) If f = 1, then this is a + b*theta (a spiral)
                  
        f_name -- This should be a + b * f(n * theta) if a,b,n,f are given. This 
                  is here to allow the user to define more general polar functions.
        f_type -- 'circle', 'lemniscate', ... This _should_ be included,
                  but if it is not we wll try to guess.
        
    Usage: 
    
        r = PolarFunction(a = 0, b = 3, n = 2, f = 'cos', 'lemniscate')
        
        This will produce an object corresponding to r^2 = 3 * cos(2 * theta)
    
        Can be shortened to: 
           r = PolarFunction(0, 3, 2, 'cos', 'lemniscate')
    
        f_name can be used as:
           r = PolarFunction(f_name = '3*sin(3 + 2*theta) + 4*cos(theta)*sin(theta)')
        
    """
        
    cache = set()
    
    # The known types 'other' is omitted here.  
    TYPES = ['circle',
        'cardioid',
        '(inner loop) limacon',
        '(convex one-loop) limacon',
        '(dimpled one-loop) limacon',
        'lemniscate',
        'rose',
        'line',
        'line through the origin',
        'circle at origin']
    
    def __init__(self, a=None, b=None, n=None, f=None,
                 f_type=None, f_name=None):
        
        if {a, b, n, f, f_name} == {None}:
            a, b, n, f, f_type = self.gen_random(f_type = f_type)  
            
        if {a, b, n, f} == {None}:
            a, b, n, f, x = self.parse_name(f_name)
            
        
        # Try to parse name if f_name is given   
        
        if a is None:
            a = 0
        if b is None: 
            b = 0
        if n is None:
            n = 1
        if f is None:
            f = 0
        
        ident = make_func('x', func_params=('x'), func_type='sympy')
        
        self.a = ident(a)
        self.b = ident(b)
        self.n = ident(n)
        self.f = sym.sympify(f)

        self.call_type = 'unknown'

        self.f_type = self.get_f_type(f_type)

        if f_name is None:
            if self.f_type == 'line through the origin':
                self.f_name = str(ident(a))
            elif self.f_type == 'circle at origin':
                self.f_name = str(ident(a))
            elif self.f_type == 'spiral':
                self.f_name = str(sym.sympify("%s + %s * theta" % (ident(a), ident(b))))
            else:
                self.f_name = str(sym.sympify("%s + %s * %s(%s * theta)" \
                                              % (ident(a), ident(b), f, ident(n))))
        else:
            self.f_name = str(sym.sympify(f_name))

            
        if self.f_type.find('lima') != -1:
            self.url = quote_plus('limacon' + self.f_name)
        elif self.f_type.find('line') != -1:
            self.url = quote_plus('line' + self.f_name)
        elif self.f_type.find('line') != -1:
            self.url = quote_plus('circle' + self.f_name)
        else:
            self.url = quote_plus(self.f_type + self.f_name)
            
        self.url = self.url
        
        self.hash_ = 17
        PolarFunction.cache.add(self)
        
    def parse_name(self, nm, G = None, A = None, B = None, N = None, F = None, T = None):
        """
        This will assume that the f_name is given in the form "a + bf(n*theta)",
        in partcular the "n*theta" order is important
    
        Parameters:
        ----------
            
        nm : A function written as a string or sympy. If t has the form 
             a + b*f(n*x) this will return a,b,n,f,x
        """
        
        
        def error():
            print("Original is not in the form \"a + b * f(n * theta)\"", file=sys.stderr)
            return (None, None, None, None)
        
        nm = sym.sympify(nm)
        
        
        if A == None:
            if str(type(nm)).find('core.add') != -1:
                if len(nm.args) > 2:
                    return error()
                A, G  = nm.args
            else:
                A = 0
                G = nm
            return self.parse_name(nm, G, A = A, B = B, N = N, F = F, T = None)
                
        elif B == None:
            if str(type(G)).find('core.mul') != -1:
                if len(G.args) > 2:
                    return error()
                B, G = G.args
            else:
                B = 1
            return self.parse_name(nm, G, A = A, B = B, N = N, F = F, T = None)
        
        elif N == None:
            F = G.func
            ag = G.args[0]
            if str(type(ag)).find('core.mul') != -1:
                if len(G.args) > 2:
                    return error()
                N, T = ag.args
            else:
                N = 1
                T = ag
            return self.parse_name(nm, G, A = A, B = B, N = N, F = F, T = T)
        
        else:
            exp = sym.sympify(A + B*F(N * T))
        
            print(exp)
            print(nm)
        
            if exp == nm:
                return (A, B, N, F, T)
            else:
                return error()
        
            return (A, B, N, F, T)

       
    def gen_random(self, f_type = None):
        """
        Generate a random f_type of polar function
        
        Parameters:
        ----------
        
        f_type: If None, first generate a random f_type.
        """
        
        if f_type is None:
            f_type = random.choice(PolarFunction.TYPES)
        
        pm = random.choice([-1,1])
        pm1 = random.choice([-1,1])        
        
        if f_type == 'circle': 
            n = 1
            a = 0
            b = pm * random.randint(1, 7)
            f = random.choice(['sin', 'cos'])
        elif f_type == 'cardioid':
            n = 1
            a = pm1 * random.randint(1, 7)
            b = pm * a
            f = random.choice(['sin', 'cos'])
        elif f_type == '(convex one-loop) limacon':
            n = 1
            a = 0
            b = random.randint(1, 4)
            a = pm * random.randint(2 * b, 3 * b)
            b = pm1 * b
            f = random.choice(['sin', 'cos'])
        elif f_type == '(dimpled one-loop) limacon':
            n = 1
            b = random.randint(2, 5)
            a = pm * random.randint(b + 1, 2 * b - 1)
            b = pm * b
            f = random.choice(['sin', 'cos'])
        elif f_type == '(inner loop) limacon':
            n = 1
            a = random.randint(1, 5)
            b = pm * random.randint(a + 1, a +5)
            a = pm * a
            f = random.choice(['sin', 'cos'])
        elif f_type == 'rose':
            a = 0
            b = pm * random.randint(1, 7)
            n = random.randint(2, 5)
            f = random.choice(['sin', 'cos'])
        elif f_type == 'lemniscate':
            n = 2
            a = 0
            b = pm * random.choice([a ** 2 for a in range(2, 7)])
            f = random.choice(['sin', 'cos'])
        elif f_type == 'line':
            n = 1
            a = 0
            b = pm * random.randint(1,6)
            f = random.choice(['sec', 'csc'])
        elif f_type == 'line through the origin':
            n = 1
            f = 0
            b = 0
            a = random.choice(['pi / 12 * %s' % i for i in range(22)])
        elif f_type == 'circle at origin':
            n = 1
            f = 0
            b = 0
            a = pm * random.randint(1,7)
        elif f_type == 'spiral':
            n = 0
            f = 1
            b = pm * random.randint(1, 4)
            a = 0
        else:
            #add more
            pass           
        
        return (a, b, n, f, f_type)
        
    def get_f_type(self, f_type=None):
        """
        Try to guess what common polar curve this is. This is not perfect! 
        It is best to prescribe the f_type.
        """
        if f_type != None:  # Assume user knows what they are doing
            return f_type
        elif self.b == 0: 
            return 'circle at origin'
        elif self.f == 0:  # Ths is a hack
            return 'line through the origin'  # This will be theta = a
        elif self.f == 1:  # This is a hack
            return 'spiral'  # This will be a + b * theta
        elif str(self.f) in ['sin', 'cos'] and self.n == 1:   
            if self.a == 0:
                return 'circle'
            elif self.a / np.abs(self.b) == 1:
                return 'cardioid'
            elif np.abs(self.a) / np.abs(self.b) < 1:
                return '(inner loop) limacon'
            elif np.abs(self.a) / np.abs(self.b) >= 2:
                return '(convex one-loop) limacon'
            else:
                return '(dimpled one-loop) limacon'
        elif self.a == 0 and str(self.f) in ['sin', 'cos'] and self.n > 1:
                return 'rose'  # Assume rose for b*sin(2*theta) unless told otherwise
        elif str(self.f) in ['sec', 'csc'] and self.a == 0:
            return 'line'
        else:
            return 'other'
            
    
    # Note this is a place Python3 would be better, I could use 
    # __call__(self, *arga, func_type = None)
    def __call__(self, *input_, **kwargs):
        """
        Named Parameters:
        
            func_type = 'numpy' (or 'sympy') 
            
            For now when a string or sympy expresion is the argument, the output is a sympy expresion. 
            It might be best to out put a string if the input is a string?
            
            Examples:
            
            f1 = PolarFunction(a = 3, b = 3, n = 1, f ='cos')
            
            f1(np.pi/3) = 4.50000000
            f1('pi/3') = 9/2 (this is a sympy expr)
            f1(sym.pi/3) = 9/2 (this is a sympy expr)
        """
           
        if 'func_type' not in kwargs:
            func_type = None
        else:
            func_type = kwargs['func_type']
        
        if func_type is None:
            is_np = False
            for i in input_:
                if type(i) is np.ndarray and i.dtype.type is np.float_:
                    is_np = is_np | True
            if is_np:
                func_type = 'numpy'
            else:
                func_type = 'sympy'
          
        self.call_type = func_type
        
        # The following is just to deal with the fact that numpy does not define sec/csc
        # See self.f_type == 'line' case below.
        p = re.compile('(sec)\\(([^)]*)\\)|(csc)\\(([^)]*)\\)')
    
        def rep(m):
            if m.group(1) == 'sec':
                return "(1/cos(" + str(m.group(2)) + "))"
            if m.group(3) == 'csc':
                return "(1/sin(" + str(m.group(4)) + "))"

        # A hack for lemniscate: return the principle root of b * cos(2 * theta) 
        # when positive. There is really a two values "function" here so some 
        # hack is necessary.
        if self.f_type == 'lemniscate':
            if func_type == 'numpy':
                f_name_ = "sqrt((%s * %s(%s * theta) > 0) * %s * %s(%s * theta))" \
                    % tuple(map(str, [self.b, self.f, self.n, self.b, 
                                     self.f, self.n]))
            else:
                f_name_ = 'Piecewise((sqrt(%s * %s(%s * theta)), %s * %s(%s * theta) >= 0))' \
                    % tuple(map(str, [self.b, self.f, self.n, self.b, 
                                     self.f, self.n]))
        # theta = a
        elif self.f_type == 'line through the origin':
            f_name_ = 'r'
        
        # sec/csc are not in numpy! Needs fixin! 
        elif func_type == 'numpy': 
            f_name_ = p.sub(rep, self.f_name)
        else:
            f_name_ = self.f_name
        
        
        if f_name_ != 'r':
            f = make_func(f_name_, func_params=('theta',), func_type=func_type)
            return f(*input_)
        else:
            return input_
        
    def __str__(self):
        if self.f_type == 'lemniscate':
            return "r**2 = " + str(sym.sympify(self.f_name))
        if self.f_type == 'line through the origin':
            return "theta = %s" % (str(sym.sympify(self.a)))
        else:
            return "r = " + str(sym.sympify(self.f_name))
        
    def __eq__(self, other):
        """
        f == g is true if they have the same graphs
        """
        
        
        if self.__class__ != other.__class__:
            return False
            
        if self.f_type == other.f_type \
            and 'other' not in [self.f_type, other.f_type]:
            # For simple cases do something simple
            if self.f_type == 'lemniscate':
                return self.f == other.f and self.b == other.b \
                    and (self.a == other.a or self.a == -other.a)
            elif self.f_type == 'circle':
                return self.f == other.f and self.b == other.b
            elif self.f_type == 'line through the origin':
                return self.a == other.a or self.a == -other.a
            elif self.f_type == 'line':
                return self.f == other.f and self.b == other.b
            elif self.f_type == 'circle at origin':
                return self.a == other.a or self.a == - other.a
            elif self.f_type in ['cardioid', '(inner loop) limacon',\
            '(convex one-loop) limacon', '(dimpled one-loop) limacon']:
                return self.f == other.f and self.b == other.b
            elif self.f_type == 'rose':
                if self.n % 2 == 0 and other.n % 2 == 0:
                    return self.f == other.f and self.n == other.n
                elif self.n % 2 == 1 and other.n % 2 == 1:
                    return self.f == other.f and self.n == other.n \
                            and self.b == other.b
                else:
                    return False
        elif self.f_type != other.f_type \
             and 'other' not in [self.f_type, other.f_type]:
            return False
        else:
            return False
            
        # Id we are some other thing try to figure out if the
        # graphs are the same.
        thetas = np.append(np.arange(0,2*np.pi, .005), 
                           np.arange(0, 2*np.pi, .005) + np.pi/2)
        X_self, Y_self = self.to_rect(thetas)
        X_other, Y_other = other.to_rect(thetas)
       
            
        D = np.array([np.min((X_other - X_self[i])**2 + 
            (Y_other - Y_self[i])**2) for i in range(X_self.size)])
            
        I = D < .01
        print(np.max(D))
        return np.all(I)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    # A cached hash
    def __hash__(self):
        if self.hash_ == 17:
            self.hash_ += 31 * hash(self.f_name)
        return self.hash_
    
    def latex(self):
        str1, str2 = str(self).split('=')
        return "%s = %s" % (sym.latex(sym.sympify(str1)), sym.latex(sym.sympify(str2)))
        
   
    def to_rect(self, thetas):
        # These are use to convert (r, theta) -> (x, y)
        x_vals = make_func('r * cos(theta)', func_params=('r', 'theta'))
        y_vals = make_func('r * sin(theta)', func_params=('r', 'theta'))
        x_vals_ = make_func('r * cos(theta)', func_params=('r', 'theta'), func_type='sympy')
        y_vals_ = make_func('r * sin(theta)', func_params=('r', 'theta'), func_type='sympy')
        
        R_ = self(thetas)
        if self.call_type == 'numpy':
            X = x_vals(R_, thetas)
            Y = y_vals(R_,thetas)
        else:
            X = x_vals_(R_, thetas)
            Y = y_vals_(R_,thetas)
        return (X, Y) 
    
    def r2d(self, t, radians=False):
        """
        Converts radians to degrees nicely for use in plots, explanatons, etc.
        Default is to convert.
        """
        if not radians:
            return sym.latex(sym.sympify('180/pi * (' + t + ')')) + "^\circ"
        else:
            return sym.latex(sym.sympify(t))
        
    def mod2pi(self, x, num_pi=2, upper=False):
        """
        This returns the modulus of a radian angle by 2*pi (actually num_pi*pi)

        Example:

        mod2pi('7*pi/3') = 'pi/3' (a string)
        mod2pi(7*sym.pi/3) = pi/3 (sympy expr equivalent to sym.pi/3)
        mod2pi(7*np.pi/3) = 1.0471975511965974 (np.pi/3)
        
        This works for negative angles as well.
        
        We also need a mod1pi function which I will incorporate through the variable
        
        Named Parameters:
            num_pi  -- Take modulo by other multiples of pi
            upper   -- Take the upper end of the interval (e.g., 2*pi instead of 0) 
        
        
        """
        
        if type(x) == str:
            return str(self.mod2pi(sym.sympify(x), num_pi=num_pi, upper=upper))
        elif str(type(x)).find('sym') != -1:
            if x == num_pi * sym.pi and upper:
                return x
            if x < 0:
                ret = num_pi * sym.pi - (sym.Abs(x) - int(sym.Abs(x) / (num_pi * sym.pi)) * num_pi * sym.pi)
            else:
                ret = x - int(x / (num_pi * sym.pi)) * num_pi * sym.pi
        else:  # type(x) is float:
            if x == num_pi * np.pi and upper:
                return x
            if x < 0:
                ret = num_pi * np.pi - (np.abs(x) - int(np.abs(x) / (num_pi * np.pi)) * num_pi * np.pi)
            else:
                ret = x - int(x / (num_pi * np.pi)) * num_pi * np.pi
        return ret
    
    def show(self, theta_min=0, theta_max=2 * np.pi,
             d_theta='2 * pi/1000', rad=False,
             r_min=0, r_max=None,
             theta_ticks=None,
             r_ticks=None, r_ticks_angle='pi / 4',
             points=[], point_labels=[], extra_points=[],
             img_type='png', file_name=None, path=".", label=False, draw_rect=False,
             xkcd=False, coloring=False, force=False):
        """
        This will draw a polar plot of the function given by r_str, e.g. r_str = '2*cos(theta)'.

        Parameters:
        ----------
            theta_min,    : The range of theta for wwhich the plot is made
            theta_max
            r_min,        : The range of radius
            r_max

            theta_ticks   : This determines the radial lines as well as labels to add. These should 
                             be strings in radians, conversion to degree will happen if rad = False
            r_ticks       : list of pairs (r,s) where r is the location ans s is the value 
            r_ticks_angle : The angle to put the r_ticks on.
            rad           : If true use radian
            d_theta       : This determines how finely to partition [0,2*pi]
            points        : This adds some points without labels to the graph
            extra_points  : These are (x,y) pairs for points not on the plot.
            point_labels  : Add points with labels to the graph. 

            img_type      :
            file_name     : Name  of image. If 'show', then just display.
            path          : Path to store image. The folder is created if does not exist
            label         : Controls whether or not to add label
            draw_rect     : Draw r = f(theta) in rectangular (theta, r) plane
            xkcd          : (True/False) Interesting rendering
            coloring      : (Boolean) If True color the graph by radius (for explanations)
            force         : (Boolean) Force rbuild of images if they exist.
            include_image : (Boolean) Format output for preview

        * = manditory
        """
        
        
        
        r_str = self.f_name
        
        if rad:
            rad_ = 'rad_'
        else:
            rad_ = ""  
        
        if file_name is None:
            file_name_ = rad_ + self.url
        else:
            file_name_ = file_name
    
        fname = path + "/" + file_name_ + "." + img_type
        if os.path.isfile(fname) and not force and file_name != 'show':
            print("The file \'" + fname + "\' exists, \
            not regenerating. Delete file to force regeneration.", file=sys.stderr)
            return fname.replace('%2', '%252')
            
        
        # r, theta = sym.symbols('r, theta')
        Thetas = np.arange(float(sym.sympify(theta_min)), float(sym.sympify(theta_max)),
                           float(sym.sympify(d_theta)))
        thetas_ = np.arange(0, 2 * np.pi, 2 * np.pi / 200)
        
        # These are use to convert (r, theta) -> (x, y)
        x_vals = make_func('r * cos(theta)', func_params=('r', 'theta'))
        y_vals = make_func('r * sin(theta)', func_params=('r', 'theta'))
        x_vals_ = make_func('r * cos(theta)', func_params=('r', 'theta'), func_type='sympy')
        y_vals_ = make_func('r * sin(theta)', func_params=('r', 'theta'), func_type='sympy')
        
        if self.f_type == 'line through the origin':
            Thetas = np.ones(1000) * float(self.a)
            R = np.linspace(-10, 10, 1000)
        else:
            R = self(Thetas)
       
        
        if r_max is not None:
            pass
        
        elif self.f_type == 'line': 
            
            r_max = 4 * int(np.abs(float(self.b)))  # For this we don't need plots with r > 16
            
        elif self.f_type == 'line through the origin':
            
            r_max = 10  # Needs to match up with circle case above
            
        else:
        
            r_max = int(np.max(np.abs(R))) + 1
        
        
        
        # Here is the data for plotting
        I = np.abs(R) <= r_max
        Thetas = Thetas[I]
        R = R[I]
        
        I_minus = R < 0
        I_plus = R > 0
        R_minus = R[I_minus]
        R_plus = R[I_plus]
        Thetas_minus = Thetas[I_minus]
        Thetas_plus = Thetas[I_plus]
    
        X = R * np.cos(Thetas)
        Y = R * np.sin(Thetas)

        # To get colors
        X_plus = X[I_plus]
        X_minus = X[I_minus]
        Y_plus = Y[I_plus]
        Y_minus = Y[I_minus]
        X_ = [X_minus, X_plus]
        Y_ = [Y_minus, Y_plus]
        R_ = [R_minus, R_plus]
        Thetas_ = [Thetas_minus, Thetas_plus]
        
        if xkcd:
            plt.xkcd()
        
        if draw_rect:
            
            fig = plt.figure(figsize=(10, 5))
            # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
         
            ax = fig.add_subplot(122, axisbg='grey')
            ax.set_axis_off()
            # plt.xlim([0,2*np.pi])
            ax1 = fig.add_subplot(121, sharey=ax)
            ax1.set_axis_on()
            ax.set_aspect('equal')
            ax.set_title('Polar Plot')
            ax1.set_title('Rectangular Plot')
            ax1.xaxis.set_ticks_position('none')
            ax1.yaxis.set_ticks_position('none') 
        else:
            
            fig = plt.figure(figsize=(5, 5))
            # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1
            ax = fig.add_subplot(111, axisbg='grey')
            plt.axis('off')
            ax.set_aspect('equal')

        # Make plot sufficiently large
        bdry = [-1.3 * r_max, 1.3 * r_max]
        ax.plot(bdry, bdry, alpha=0.0)

        ax.set_xticks([])
        ax.set_yticks([])
       

        # Set up a white circular plotting area
        ax.fill_between(x_vals(1.1 * int(r_max), thetas_), y_vals(1.1 * int(r_max), thetas_),
                        - y_vals(1.1 * int(r_max), thetas_), color='white')
        
         # Set some default positioning of the r_ticks
        if self.b > 0 and str(self.f) in ['cos', 'sec']:
                case = '0'
        elif self.b > 0 and str(self.f) in ['sin', 'csc']:
            case = 'pi/2'
        elif self.b < 0 and str(self.f) in ['cos', 'sec']:
            case = 'pi'
        else:
            case = '3*pi/2'

        r_ticks_angle = 5 * sym.pi / 6 + sym.sympify(case)
        
        if self.f_type == 'lemniscate':
            
            if self.b > 0 and str(self.f) == 'cos':
                case = '0'
            elif self.b > 0 and str(self.f) == 'sin':
                case = 'pi/4'
            elif self.b < 0 and str(self.f) == 'cos':
                case = 'pi/2'
            else:
                case = '3*pi/4'
        
            r_ticks_angle = 3 * sym.pi / 4 + sym.sympify(case)
            
        elif self.f_type == 'circle':
            
            r_ticks_angle = sym.pi / 2 + sym.sympify(case)
            
        elif self.f_type == 'line':
    
            r_ticks_angle = 2 * sym.pi / 3 + sym.sympify(case)
        
        else:
            
            pass
        
            
        if r_ticks == None:
            d_r = int(np.sqrt(r_max))
            r_ticks = [(i * d_r, r_ticks_angle, i * d_r) for i in range(int(r_max / d_r) + 1)]
            
      
        r_ticks_ = [(x_vals_(tck[0], tck[1]), y_vals_(tck[0], tck[1]), sym.latex(sym.sympify(tck[2])))
                       for tck in r_ticks]
                    
        
        

        # Set up grid 
        # First the circles
        [ax.plot(x_vals_(a[0], thetas_), y_vals_(a[0], thetas_), color='grey',
                     linestyle='-', alpha=0.1) for a in r_ticks]
        

        # Put Labels on ticks
        [ax.text(tick[0], tick[1], tick[2],
                 ha='center', va='center', size=10) for tick in r_ticks_[1:]]

        
        
        # Next the radiating lines
        if theta_ticks == None:
            theta_ticks = ['%s*pi/6' % i for i in range(12)]
           
        
        
        [ax.plot([1.05 * x_vals_(int(r_max), theta), -1.05 * x_vals_(int(r_max) , theta)],
                 [1.05 * y_vals_(int(r_max), theta), -1.05 * y_vals_(int(r_max), theta)],
                 color='grey', linestyle='-', alpha=0.1) for theta in theta_ticks]

        # Add text to theta_ticks

        
        [ax.text(1.15 * x_vals_(r_max, s), 1.2 * y_vals_(r_max, s),
                '$%s$' % (self.r2d(s, radians=rad)), ha='center', va='center', alpha=0.6, weight='bold',
                rotation=int(sym.sympify(s + "* 180/pi").evalf()) - 90, size=12) 
                    for s in theta_ticks] 
        
        # If including rectangular plot, set that up.
        
        if draw_rect:
            
            # Annoying case!    
            theta_ticks_ = theta_ticks
            
            if self.f_type == 'spiral':
                theta_ticks_ = ['%s*pi/3' % i for i in range(12)]
            
            theta_ticks_ = [ sym.sympify(u) + sym.sympify(theta_min) for u in theta_ticks_]
            theta_ticks_ = [str(u) for u in theta_ticks_ 
                                if u <= sym.sympify(theta_max) and u >= sym.sympify(theta_min)]
            
            
            r_ticks_y = [tck[0] for tck in r_ticks] + [-tck[0] for tck in r_ticks]
            ax1.set_yticks(r_ticks_y)
            ax1.set_xticks(map(lambda x: float(sym.sympify(x)), theta_ticks_))
            
            # make room for theta in degrees
            if rad:
                rotation = 0
            else:
                rotation = -45
            ax1.set_xticklabels(map(lambda x: '$%s$' % self.r2d(x, radians=rad), theta_ticks_), rotation=rotation)
            [ax1.plot([t, t], [-1.2 * r_max, 1.2 * r_max], color='grey', alpha=0.1) 
                     for t in map(lambda x: float(sym.sympify(x)), theta_ticks_)]
                                  
            [ax1.plot([float(sym.sympify(theta_min)) - .1, float(sym.sympify(theta_max)) + .1],
                             [y, y], color='grey', alpha=0.1) for y in r_ticks_y]
            
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            
       
        
        # Plot some points and label
        labels = {}
        labels_rect = {}
        pts = [sym.sympify(pt) for pt in point_labels if sym.sympify(pt) >= sym.sympify(theta_min) 
                     and sym.sympify(pt) <= sym.sympify(theta_max)] 
        for pt in pts:
            r_ = self(pt)
            if r_ is None or np.imag(r_) != 0:
                continue
            
            if self.f_type == 'lemniscate':
                rs = [r_, -r_]
            else:
                rs = [r_]
            
            for q in rs:
                coord_rect = (float(sym.sympify(pt)), q)
                coord = (x_vals(q, pt), y_vals(q, pt))
                lbl = "$%s$" % (self.r2d(str(pt), rad))
            
                if coord not in labels:
                    labels[coord] = {lbl}
                else:
                    labels[coord].add(lbl)
                    
                if coord_rect not in labels_rect:
                    labels_rect[coord_rect] = {lbl}
                else:
                    labels_rect[coord_rect].add(lbl)
    
        def lh(c):
            return np.sqrt(c[0] ** 2 + c[1] ** 2)
        
        def pos(coord):
            d = {}
            if coord[0] < 0:
                d['ha'] = 'right'
            else:
                d['ha'] = 'left'
            if coord[1] < 0:
                d['va'] = 'top'
            else:
                d['va'] = 'bottom'
            return d
        
        def pos_rect(coord_rect):
            d = {}
            if coord_rect[1] < 0:
                d['va'] = 'bottom'
            else: 
                d['va'] = 'top'
            return d
        
        if len(point_labels) > 0:
            [ax.text(coord[0] * (1 + .05 * r_max / lh(coord)), coord[1] * (1 + 0.05 * r_max / lh(coord)),
                     "%s" % (','.join(labels[coord])),
                     ha=pos(coord)['ha'], va=pos(coord)['va'], size=12, weight='bold', zorder=10) 
                 for coord in labels if np.abs(coord[0]) + np.abs(coord[1]) > 1.0e-6]
            
            [ax.text(0.05 * r_max, -0.05 * r_max, "%s" % (','.join(labels[coord])),
                 ha='left', va='top', size=12, weight='bold', zorder=10) for 
             coord in labels if np.abs(coord[0]) + np.abs(coord[1]) <= 1.0e-6]
            
            if draw_rect:
                [ax1.text(coord_rect[0] , coord_rect[1],
                     "%s" % (','.join(labels_rect[coord_rect])),
                     ha='center', va=pos_rect(coord_rect)['va'], size=12,
                          weight='bold', zorder=10) for coord_rect in labels_rect]
            
           
        # add points to labels
        [ax.plot(x_vals_(self(pt), pt), y_vals_(self(pt), pt), "ro", zorder=10, markersize=3) 
             for pt in point_labels if self(pt) is not None 
                 and sym.sympify(pt) >= sym.sympify(theta_min) 
                 and sym.sympify(pt) <= sym.sympify(theta_max)] 
        
        if draw_rect:
            [ax1.plot(float(sym.sympify(pt)), self(pt), "ro", zorder=10, markersize=3) 
             for pt in point_labels if self(pt) is not None 
                 and sym.sympify(pt) >= sym.sympify(theta_min) 
                 and sym.sympify(pt) <= sym.sympify(theta_max)] 
            

        if self.f_type == 'lemniscate':
            [ax.plot(x_vals_(-self(pt), pt), y_vals_(-self(pt), pt), "ro", zorder=10, markersize=3) 
                 for pt in point_labels if self(pt) is not None 
                     and sym.sympify(pt) >= sym.sympify(theta_min) 
                     and sym.sympify(pt) <= sym.sympify(theta_max)]  
            
            if draw_rect:
                [ax1.plot(float(sym.sympify(pt)), -self(pt), "ro", zorder=10, markersize=3) 
                 for pt in point_labels if self(pt) is not None 
                     and sym.sympify(pt) >= sym.sympify(theta_min) 
                     and sym.sympify(pt) <= sym.sympify(theta_max)] 
        
        # Add extra points without labels
        
        [ax.plot(x_vals_(self(pt), pt), y_vals_(self(pt), pt), "bo", zorder=10, markersize=3) 
             for pt in points if self(pt) is not None 
                 and sym.sympify(pt) >= sym.sympify(theta_min) 
                 and sym.sympify(pt) <= sym.sympify(theta_max)]  

        if self.f_type == 'lemniscate':
            [ax.plot(x_vals_(-self(pt), pt), y_vals_(-self(pt), pt), "ro", zorder=10, markersize=3) 
                 for pt in points if self(pt) is not None 
                     and sym.sympify(pt) >= sym.sympify(theta_min) 
                     and sym.sympify(pt) <= sym.sympify(theta_max)]  
            
        # Need to occasionally add some extra points that are not on the graph
        [ax.plot(pt[0], pt[1], "bo", zorder=10, markersize=3) 
             for pt in extra_points]
        
        # Plot the actual graph
        if coloring:
            colors = [[((150 + 1.0 * z / r_max * 100) / 255.0, .7 * (150 + 1.0 * z / r_max * 100) / 255.0,
                        (150 - 1.0 * z / r_max * 100) / 255.0) for z in R_[i]] for i in range(2)]
        else:
            colors = [[(.3, .6, .3, .5)], [(.3, .6, .3, .5)]]
       
        for i in range(2):
                
            ax.scatter(X_[i], Y_[i], c=colors[i],
                       alpha=0.5, zorder=7, marker='o', edgecolors='face', s=2)
            if self.f_type == 'lemniscate':
                ax.scatter(-X_[1 - i], -Y_[1 - i], c=colors[1 - i],
                           alpha=0.5, zorder=7, edgecolors='face', s=2)
            

            if label:
                ax.legend([r"$%s$" % (self.latex())], loc='lower center', bbox_to_anchor=(0.0, -0.15),
                          frameon=False, framealpha=0.0)
            if draw_rect:
                ax1.scatter(Thetas_[i], R_[i], c=colors[i], marker='o',
                            alpha=0.5, edgecolors='face', s=2)
                if self.f_type == 'lemniscate':
                    ax1.scatter(Thetas_[1 - i], -R_[1 - i], c=colors[1 - i],
                                marker='o', alpha=0.5, edgecolors='face', s=2)
        
        # plt.axis('off')

        if file_name == 'show':
            plt.show()
            plt.close()
        else:
            tools.make_folder_if_necessary(".", path)
            plt.savefig(fname) 
            plt.close()
            
        
        return fname.replace('%2', '%252') 
        
    def explain(self, rad=False, path="explanations", include_image=False, 
                xkcd=False, force=False):
        """
        For this to work well, the parameters a,b,n,f _probably need to be used for now.
        
        Parameters:
        ----------
            path            : (String) This is where images will be stored.
            rad             : (Boolean) Use radians (True) / degrees (False)
            include_image   : (Boolean) For testing, include <img href = "..."> in output. If False output
                              ${path/f.url.png}$
            force           : force rebuilding of existing images
            xkcd            : Turn on XKCD Artist
        """
        
        a_ = float(self.a)
        b_ = float(self.b)
        n_ = float(self.n)
        
        case = '0'
        
        if self.b >= 0 and str(self.f) in ['cos', 'sec'] :
            case = '0'
            even_odd = 'odd'
            vert_horiz = 'vertical'
        elif self.b >= 0 and str(self.f) in ['sin', 'csc']:
            case = 'pi/2'
            even_odd = 'even'
            vert_horiz = 'horizontal'
        elif self.b < 0 and str(self.f) in ['cos', 'sec']:
            if str(self.f) == 'cos':
                case = 'pi'
            else:
                case = '0'
            even_odd = 'odd'
            vert_horiz = 'vertical'
        elif self.b < 0 and str(self.f) in ['sin', 'csc']:
            if str(self.f) == 'sin':
                case = '3*pi/2'
            else:
                case = 'pi/2'
            even_odd = 'even'
            vert_horiz = 'horizontal'
            
        default_points = []
        extra_points = []
        default_point_labels = []
        
        theta_min, theta_max = ['0', '2*pi']
        
        
        
        
        x_vals_ = make_func('r * cos(theta)', func_params=('r', 'theta'), func_type='sympy')
        y_vals_ = make_func('r * sin(theta)', func_params=('r', 'theta'), func_type='sympy')
        
        note = "Note: The points on the graph are labeled \
                with the value of $_\\theta$_ which gives rise to that point, in the case that \
                $_r(\\theta)<0$_, this might appear strange at first."
        
        
        if rad:
            rad_ = 'rad_'
        else:
            rad_ = ""  
        
        default_file_name = rad_ + self.url
        
        if include_image:
            image_name = "<img width = 50%s src=\'%s/%s%s.png\'>" \
                    % ('%', path, rad_, self.url.replace('%2', '%252'))
        else:
            image_name = "${%s/%s%s.png}$" % (path, rad_, self.url.replace('%2', '%252'))

        # The theta at which a limacon is maximal distance from the pole:
        lim_max = self.mod2pi(sym.pi * sym.sign(self.a) + sym.sympify(case), upper=True)
        lim_min = str(self.mod2pi(lim_max + sym.pi))
        lim_max = str(lim_max)
        
        # On graphs of type 'other' just draw some test points and labe the graph. For specific 
        # named graphs we can do better.
        if self.f_type == 'other':
            default_points = [sym.pi / 3 * i for i in range(6)] 
            
        
        elif self.f_type == 'circle at origin':
            
            explanation = """
            The graph of $_%s$_ is a circle with center at the origin and radius $_%s$_. 
            The graph is symetric about $_\\theta = %s$_ ($_y$_-axis), 
            polar axis ($_x$_-axis), and the pole (origin). 
            <br>
            %s
            <br>
            %s
            """ % (self.latex(), self.r2d('pi/2', rad), self.a, note, image_name)
            
            default_points = [sym.pi / 3 * i for i in range(6)]
            
            
        elif self.f_type == 'lemniscate':
            
            if self.b > 0 and str(self.f) == 'cos':
                case = '0'
            elif self.b > 0 and str(self.f) == 'sin':
                case = 'pi/4'
            elif self.b < 0 and str(self.f) == 'cos':
                case = 'pi/2'
            else:
                case = '3*pi/4'
                
            # The domain
                
            theta_min, theta_max = map(lambda x : "%s - pi/4" \
                % self.mod2pi(x, num_pi=2, upper = True),
                                       ['0 + %s' % case, 'pi/2 + %s' % case])
                                       
            
            exp_string = """
                    The graph of $_%s$_ is a lemniscate symetric about the line $_\\theta =
                    %s$_. The natural domain on which this graph is defined is 
                    $_[%s, %s]$_.The graph is furthest away from the pole at $_\\theta = %s$_ with $_r=%s$_ 
                    and $_r=%s$_. The graph is at the pole 
                    at $_\\theta = %s$_ and $_\\theta = %s$_. 
                    <br>
                    %s
                    <br>
                    %s
                    """
                  
                
            explanation = exp_string % (self.latex(),
                        self.r2d('0 + %s' % case, rad),
                        self.r2d(theta_min, rad), self.r2d(theta_max, rad), self.r2d('0 + %s' % case, rad),
                        sym.latex(sym.sympify(sym.sqrt(sym.Abs(self.b)))),
                        sym.latex(sym.sympify(-sym.Abs(sym.sqrt(self.b)))),
                        self.r2d('-pi/4 + %s' % case, rad), self.r2d('pi/4 + %s' % case, rad), note, image_name)

            default_point_labels = ['0 + %s' % (case), 'pi/4 + %s' % (case), '-pi/4 + %s' % (case)]
            
            
            
            
        elif self.f_type == 'cardioid':
            
            
            exp_string = """
                    The graph of $_%s$_ is a cardioid symetric about the line $_\\theta =
                    %s$_. The natural domain on which this graph is defined is 
                    $_[%s, %s]$_.The graph is furthest away from the pole at $_\\theta = %s$_ 
                    with $_r=%s$_. The graph is at the pole 
                    at $_\\theta = %s$_ and intersects the line $_\\theta = %s$_ at two places
                    The graph has horizontal / vertical tangents at
                    $_\\theta = %s, %s, %s, %s, %s$_. 
                    <br>
                    %s
                    <br>
                    %s
                    """
            
            explanation = exp_string % (self.latex(), self.r2d('0 + %s' % case, rad),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad), self.r2d(lim_max, rad),
                  sym.latex(self(lim_max)), self.r2d(lim_min, rad),
                  self.r2d('pi/2 + %s' % case),
                  self.r2d(self.mod2pi('0 + %s' % case), rad), self.r2d(self.mod2pi('pi/3 + %s' % case), rad),
                  self.r2d(self.mod2pi('2*pi/3 + %s' % case), rad), self.r2d(self.mod2pi('5*pi/3 + %s' % case), rad),
                  self.r2d(self.mod2pi('4*pi/3 + %s' % case), rad), note, image_name)
            
            default_point_labels = map(self.mod2pi, ['0 + %s' % (case), 'pi/3 + %s' % (case), \
                            'pi/2 + %s' % case, '2*pi/3 + %s' % (case), 'pi + %s' % case, \
                            '-pi/3 + %s' % (case), '-pi/2 + %s' % case, '-2*pi/3 + %s' % (case)])
            
        
        
        
        elif self.f_type == '(inner loop) limacon':
            

            exp_string = """
                    The graph of $_%s$_ is a lima&ccedil;on with an inner loop 
                    symetric about the line $_\\theta =
                    %s$_. The natural domain on which this graph is defined is 
                    $_[%s, %s]$_. The outer loop of the graph is furthest away from the pole at $_\\theta = %s$_ 
                    with $_r=%s$_. The inner loop of the graph is furthest away from the pole at 
                    $_\\theta = %s$_ with $_r = %s$_. The graph is at the pole 
                    when $_%s = %s$_ and crosses $_\\theta = %s$_ at two points. 
                    The graph has horizontal / vertical tangents at indicated points.
                    <br>
                    %s
                    <br>
                    %s
                    """

            explanation = exp_string % (self.latex(), self.r2d('0 + %s' % case, rad),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad), self.r2d(lim_max, rad),
                  sym.latex(self(lim_max)), self.r2d(lim_min, rad),
                  sym.latex(self(lim_min)),
                  sym.latex(sym.sympify('cos(theta - %s)' % case)), sym.latex(sym.sympify("-(%s)/(%s)" % (self.a, self.b))),
                  self.r2d(self.mod2pi('pi / 2 - %s' % case, num_pi=1), rad), note, image_name)

            default_point_labels = map(self.mod2pi, ['0 + %s' % (case), 'pi/2 + %s' % (case), \
                               'pi + %s' % (case), '3*pi/2 + %s' % (case)])
            
            default_points = [(-a_ + np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              (-a_ - np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              - 1.0 * a_ / (2 * b_), -1.0 * a_ / b_]
            
            default_points = np.array([d for d in default_points if np.abs(d) <= 1])
            
            if str(self.f) == 'cos':
                default_points = list(np.append(np.arccos(default_points), -np.arccos(default_points)))
            else:
                default_points = list(np.append(np.arcsin(default_points), np.pi - np.arcsin(default_points)))
            
            default_points = map(self.mod2pi, default_points)
            
        elif self.f_type == '(convex one-loop) limacon':

            exp_string = """
                    The graph of $_%s$_ is a convex one-loop lima&ccedil;on 
                    symetric about the line $_\\theta =
                    %s$_. The natural domain on which this graph is defined is 
                    $_[%s, %s]$_. The graph is furthest away from the pole at $_\\theta = %s$_ 
                    with $_r=%s$_. The graph is closest to the pole at $_\\theta = %s$_ with $_r = %s$_
                    and crosses $_\\theta = %s$_ at two points. 
                    The graph has horizontal / vertical tangents at indicated points.
                    <br>
                    %s
                    <br>
                    %s
                    """

            explanation = exp_string % (self.latex(), self.r2d('0 + %s' % case, rad),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad),
                  self.r2d(lim_max, rad), sym.latex(self(lim_max)),
                  self.r2d(lim_min, rad), sym.latex(self(lim_min)),
                  self.r2d(self.mod2pi('pi/2 + %s' % case, num_pi=1), rad),
                  note, image_name)

            default_point_labels = map(self.mod2pi, ['0 + %s' % (case), 'pi/2 + %s' % (case), \
                               'pi + %s' % (case), '3*pi/2 + %s' % (case)])
            
            default_points = [(-a_ + np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              (-a_ - np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              - 1.0 * a_ / (2 * b_), -1.0 * a_ / b_]
            
            
            default_points = np.array([d for d in default_points if np.abs(d) <= 1])
            
            if str(self.f) == 'cos':
                default_points = list(np.append(np.arccos(default_points), -np.arccos(default_points)))
            else:
                default_points = list(np.append(np.arcsin(default_points), np.pi - np.arcsin(default_points)))
            
            default_points = map(self.mod2pi, default_points)
            
            
            
        elif self.f_type == '(dimpled one-loop) limacon':

            exp_string = """
                    The graph of $_%s$_ is a convex one-loop lima&ccedil;on 
                    symetric about the line $_\\theta =
                    %s$_. The natural domain on which this graph is defined is 
                    $_[%s, %s]$_. The graph is furthest away from the pole at $_\\theta = %s$_ 
                    with $_r=%s$_. The graph is closest to the pole at $_\\theta = %s$_ with $_r = %s$_
                    and crosses $_\\theta = %s$_ at two points. 
                    The graph has horizontal / vertical tangents at indicated points
                    <br>
                    %s
                    <br>
                    %s
                    """

            explanation = exp_string % (self.latex(), self.r2d('0 + %s' % case, rad),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad),
                  self.r2d(lim_max, rad), sym.latex(self(lim_max)),
                  self.r2d(lim_min, rad), sym.latex(self(lim_min)),
                  self.r2d(self.mod2pi('pi/2 + %s' % case, num_pi=1), rad),
                  note, image_name)


            default_point_labels = map(self.mod2pi, ['0 + %s' % (case), 'pi/2 + %s' % (case), \
                               'pi + %s' % (case), '3*pi/2 + %s' % (case)])
            
            default_points = [(-a_ + np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              (-a_ - np.sqrt((a_) ** 2 + 8 * (b_) ** 2)) / (4.0 * b_),
                              - 1.0 * a_ / (2 * b_), -1.0 * a_ / b_]
            
            
            default_points = np.array([d for d in default_points if np.abs(d) <= 1])
            
            if str(self.f) == 'cos':
                default_points = list(np.append(np.arccos(default_points), -np.arccos(default_points)))
            else:
                default_points = list(np.append(np.arcsin(default_points), np.pi - np.arcsin(default_points)))
            
            default_points = map(self.mod2pi, default_points)
            
            
        elif self.f_type == 'rose':
            
            if self.n % 2 == 1:
                theta_min, theta_max = ['0', 'pi']
            
            
            exp_string = """
                    The graph of $_%s$_ is a rose with %s pedals symetric about the line $_\\theta =
                    %s$_. The natural domain on which this 
                    graph is defined is $_[%s, %s]$_. The graph is furthest away from the pole when 
                    $_%s \\cdot \\theta$_ is an integer multiple of $_%s$_ at which the distance 
                    from the pole is $_r=%s$_. The graph is at the pole when $_%s \\cdot 
                    \\theta$_ is an integer multiple of $_%s$_. 
                    <br>
                    %s
                    <br>
                    %s
                    """
        
            explanation = exp_string % (self.latex(), self.n * (1 + (self.n + 1) % 2),
                  self.r2d('0 + %s' % case, rad), self.r2d(theta_min, rad), self.r2d(theta_max, rad),
                  self.n, self.r2d(self.mod2pi('pi + %s' % case, num_pi=1), rad), sym.Abs(self.b),
                  self.n, self.r2d(self.mod2pi('pi/2 + %s' % case, num_pi=1, upper=True), rad),
                  note, image_name)
            
            default_point_labels = map(lambda x: self.mod2pi(x, num_pi=2 - self.n % 2, upper=True),
                                       ['%s  * pi / (2 * %s) + %s' % (k, self.n, case) 
                                                    for k in range(((1 + self.n) % 2 + 1) * 2 * self.n)])
            
            default_points = []
            
        elif self.f_type == 'spiral':
            
            theta_min , theta_max = ['0', 4 * sym.pi]
            
            exp_string = """
                    The graph of $_%s$_ is a spiral The natural domain on which this 
                    graph is defined is $_[%s, \infty)$_. The graph gets $_%s$_ further away from the pole 
                    on each complete revolution.
                    <br>
                    %s
                    <br>
                    %s
                    """
            
            explanation = exp_string % (self.latex(), self.r2d('0', rad), sym.latex(sym.sympify(self('2*pi'))),
                  note, image_name)
            
            default_point_labels = ['2 * pi * %s' % i for i in range(int(4 / self.b))] 
            
            
        elif self.f_type == 'circle':
            
            if even_odd == 'odd':
                theta_min, theta_max = ['-pi/2', 'pi/2'] 
            else:
                theta_min, theta_max = ['0', 'pi'] 
                
            exp_string = """
                    The graph of $_%s$_ is a circle symetric about the line $_\\theta =
                    %s$_. The natural domain on which this 
                    graph is defined is $_[%s, %s]$_. The graph is furthest away from the pole when 
                    $_\\theta = %s$_ at which the distance 
                    from the pole is the circles diameter $_%s$_. The graph is at the pole when 
                    $_\\theta$_ is an %s multiple of $_%s$_. The center of the circle is at $_(%s,%s)$_.
                    <br>
                    %s
                    <br>
                    %s
                    """
        
            explanation = exp_string % (self.latex(),
                  self.r2d(self.mod2pi('0 + %s' % case, num_pi=1), rad),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad),
                  self.r2d(self.mod2pi('0 + %s' % case, num_pi=1), rad), sym.Abs(self.b), even_odd,
                  self.r2d('pi - %s' % self.mod2pi('pi/2 + %s' % case, num_pi=1), rad),
                  sym.latex(x_vals_(self.b / 2, '0 + %s' % case)),
                  sym.latex(y_vals_(self.b / 2, '0 + %s' % case)),
                  note, image_name)
            
            default_point_labels = map(lambda x: self.mod2pi(x, num_pi=1),
                                       ['%s  * pi / 4 + %s' % (k, case)  for k in range(4)])
            
            default_points = []
            
            extra_points = [(x_vals_(sym.Abs(self.b) / 2, '0 + %s' % case), y_vals_(sym.Abs(self.b) / 2, '0 + %s' % case))]
        
        elif self.f_type == 'line':
            
            if even_odd == 'odd':
                theta_min, theta_max = ['-pi/2', 'pi/2'] 
            else:
                theta_min, theta_max = ['0', 'pi'] 
            
            
            exp_string = """
                    The graph of $_%s$_ is a %s line through $_(%s,%s)$_. The natural domain on which this 
                    graph is defined is $_(%s, %s)$_. 
                    <br>
                    %s
                    <br>
                    %s
                    """
        
            explanation = exp_string % (self.latex(), vert_horiz,
                  sym.latex(x_vals_(self.b, '0 + %s' % case)),
                  sym.latex(y_vals_(self.b, '0 + %s' % case)),
                  self.r2d(theta_min, rad), self.r2d(theta_max, rad),
                  note, image_name)
            
            default_point_labels = ["%s - pi/3" % self.mod2pi('%s  * pi / 6 + %s' % (k, case))
                                     for k in range(5)]
            
        elif self.f_type == 'line through the origin':
            
            exp_string = """
                    The graph of $_%s$_ is a line through the origin. The slope is $_\\tan\\left(%s\\right) = %s$_.
                    <br>
                    %s
                    """
        
            explanation = exp_string % (self.latex(), sym.latex(self.a), sym.latex(sym.sympify('tan(%s)' % self.a)),
                                        image_name)
                                        
        
         
            
        self.show(points=default_points, extra_points=extra_points,
                  point_labels=default_point_labels, label=True,
                  path=path, file_name=default_file_name, rad=rad,
                  theta_max=theta_max, theta_min=theta_min, draw_rect=True,
                  coloring=True, force=force)
        
        return explanation
        
            
        
if __name__ == "__main__":
#    PolarFunction(a=0, b=2, n=2, f='sin')
#    PolarFunction(a=0, b=2, n=2, f='sin', f_type='lemniscate')
#    PolarFunction(a=1, b=2, n=1, f='cos')
#    PolarFunction(a=3, b=-4, n=1, f='cos')
#    PolarFunction(a=6, b=2, n=1, f='sin')
#    PolarFunction(a=0, b=-3, n=1, f='csc')
#    PolarFunction(a='pi/4', f=0)
#    PolarFunction(a=0, b=2, f=1)
#    PolarFunction(a = 6, b = -36, n = 2, f = 'sin', f_type = 'lemniscate')
#    PolarFunction(f_type = 'line through the origin')
#    
#    
#    # Generate a bunc of random graphs of specific types
#    for ftype in PolarFunction.TYPES:
#        for i in range(6):
#            f = PolarFunction(f_type = ftype)
#            
#    # Now generate a bunch of random graphs (non-specific type)
#    for i in range(10):
#        PolarFunction()
#    
#    # Generate plots/ explanations
#    explanations = []
#    for f in PolarFunction.cache:
#        rad_ = random.choice([True,False])
#        f.show(path='test', rad = rad_)
#        explanations.append(f.explain(path='test/explanations', include_image=True,
#                              rad=rad_))
#                    
#                    
#    for explanation in explanations:
#        print(explanation)
#         print("\n\n\n<br>")
    
    # You can use either f_name    
    f = PolarFunction(f_name = '2 * sin(2*theta)')
    # or the pieces to build the name
    g = PolarFunction(a=0, b=-2, n=2, f='sin')
    print(f == g) # Same graphs
    print("g,f have the sam graph, but they are distinct objects. \
        {f,g} has size %s" % len({f,g}))
    f.show(file_name = 'show')
    g.show(file_name = 'show')