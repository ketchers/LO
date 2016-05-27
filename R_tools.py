import numpy as np
import sympy as sym


def lfy(f, p):
    """
    This is used as
    
    f = lfy('a + b*cos(theta)', ('a', 'b','theta'))
    f(2,3,sym.pi/2)
    
    Important! For single parameter functions, you need to use ('a',) ...
    
    f = lfy('cos(theta)', ('theta')) creates paraamters 't', 'h', ....
    
    You need to use 
    
    f = lfy('cos(theta)', ('theta',))
    """
    sym.var(p)
    def f_(*q):
        return sym.sympify(f).subs(zip(p,q))
    return f_

def make_func(func_desc, func_params = ('x'), func_type = 'numpy'):
    """
    This will take a string or sympy expresion for a function together with a tuple of
    parameters / vars and return a function that will correctly evaluate the expression
    in many circumstances.
    
    Param:
    
        func_desc -- A string, e.g., '3 + 2 * cos(x)', or a sympy expression: 3 + 2 * sym.cos(x)
    
    Named Parameters:
    
        func_params -- The varables in the functon: 'a + b * cos(x)' has fuc_params = ('a','b','x')
        func_type   -- 'np' (numpy / numeric) or 'sym' (sympy / sybolic)
        
        array([-1, -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5,
                2 + 3*sqrt(3)/2, 7/2, 2, 1/2, -3*sqrt(3)/2 + 2, -1,
               -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5, 2 + 3*sqrt(3)/2,
                7/2, 2, 1/2, -3*sqrt(3)/2 + 2], dtype=object)
        
        
    Examples:
    
    # Example with sympy
    
    f = make_func('a + b * cos(t)', func_params = ('a', 'b', 't'), func_type = 'sympy')
    f(2, -3, np.array([2*sym.pi*i/12for i in range(24)]))
    
    array([-1, -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5,
            2 + 3*sqrt(3)/2, 7/2, 2, 1/2, -3*sqrt(3)/2 + 2, -1,
           -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5, 2 + 3*sqrt(3)/2,
            7/2, 2, 1/2, -3*sqrt(3)/2 + 2], dtype=object)
            
    
    # Example with numpy
    
    f = make_func('a + b * cos(t)', func_params = ('a', 'b', 't'), func_type = 'numpy')
    f(2, -3, np.array([2*np.pi*i/12for i in range(24)]))
    
    array([-1.        , -0.59807621,  0.5       ,  2.        ,  3.5       ,
        4.59807621,  5.        ,  4.59807621,  3.5       ,  2.        ,
        0.5       , -0.59807621, -1.        , -0.59807621,  0.5       ,
        2.        ,  3.5       ,  4.59807621,  5.        ,  4.59807621,
        3.5       ,  2.        ,  0.5       , -0.59807621])
    
    
    # Example with func_desc a sympy expression
    
    a,b,t = sym.symbols('a,b,t')
    f = make_func(a + b * sym.cos(t), func_params = ('a', 'b', 't'), func_type = 'sympy')
    f(2, -3, np.array([2*sym.pi*i/12for i in range(24)]))
    
    array([-1, -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5,
            2 + 3*sqrt(3)/2, 7/2, 2, 1/2, -3*sqrt(3)/2 + 2, -1,
           -3*sqrt(3)/2 + 2, 1/2, 2, 7/2, 2 + 3*sqrt(3)/2, 5, 2 + 3*sqrt(3)/2,
            7/2, 2, 1/2, -3*sqrt(3)/2 + 2], dtype=object)
    """
    
    def func(*func_inputs):
        """
        This is the actual function returned. 
        
        Param: 
        
            func_inputs -- These are assumed to be numeric / string / sympy or np.ndarrays of these
       
        """
        
        if func_type == 'numpy':
            if  func_desc.__class__ is str:
                f_ = np.vectorize(sym.lambdify(func_params, sym.sympify(func_desc), func_type))
            else: # s is already a sympy expression
                f_ = np.vectorize(sym.lambdify(func_params, func_desc, func_type))
        else:
            f_ = np.vectorize(lfy(func_desc, func_params))
        
        # if a list is in the input convert to an np.ndarray
        func_inputs = map(lambda x: np.array(x) if type(x) is list else x, func_inputs)
        
        # If inputs are strings or np.arrays of strings we need to convert them to sym expresions or 
        # numpy.float64 ....
        if func_type == 'sympy':
            # convert input that s array of strings
            func_inputs = map(lambda ar: np.array(map(sym.sympify, ar)) 
                              if type(ar) is np.ndarray else ar, func_inputs) 
            # Convert inputs that are strings
            func_inputs = map(lambda s: sym.sympify(s) if type(s) is not np.ndarray else s, func_inputs)
        elif func_type == 'numpy':
            func_inputs = map(lambda ar: np.array(map(lambda x: float(sym.sympify(x)), ar)) 
                               if type(ar) is np.ndarray and ar.dtype is not np.float_ else ar, func_inputs) 
                                 #  and ar.dtype.type is np.string_ else ar, func_inputs) 
            func_inputs = map(lambda s: float(sym.sympify(s)) if type(s) is not np.ndarray else s, func_inputs) 
                                 #if type(s) is str else s, func_inputs)
       
    
        Y = f_(*func_inputs)
        
        if Y.shape == (1,):
            return Y[0]
        elif Y.shape == ():
            return np.asscalar(Y)
        else:
            return Y
        
       
    return func

if __name__ == "__main__":
    
    f = make_func('Piecewise((sqrt(4 * cos(2 * theta)), 4 * cos(2 * theta) >= 0))', 
              func_params = ('theta'), func_type = 'sympy')
    pts = ['0 + 0', 'pi/4 + 0', 'pi/2 + 0']
    
    print f(pts)
    print f([sym.pi/16 * i for i in range(-4,5)])
    print f(np.arange(-np.pi/4, np.pi/4,np.pi/16))
    print f('pi/8')
    
            
