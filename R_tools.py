from __future__ import print_function
import numpy as np
import sympy as sym
import sys
import random



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
    
    

def mod2pi(x, num_pi=2, upper=False):
        """
        This returns the modulus of a radian angle by 2*pi (actually num_pi*pi)

        Example:

        mod2pi('7*pi/3') = 'pi/3' (a string)
        mod2pi(7*sym.pi/3) = pi/3 (sympy expr equivalent to sym.pi/3)
        mod2pi(7*np.pi/3) = 1.0471975511965974 (np.pi/3)
        
        mod2pi('4*pi/3', num_pi = 1) = 'pi/3
        
        This works for negative angles as well.
        
        Named Parameters:
            
            num_pi -- Perform mod num_pi * pi
            upper  -- If true: Return 2 * pi instead of 0 for mod2pi('2*pi')
                      This is (0,2*pi] instead of [0,2*pi)
        """
        
        if type(x) == str:
            return str(mod2pi(sym.sympify(x), num_pi=num_pi, upper=upper))
        elif str(type(x)).find('sym') != -1:
            if x < 0:
                ret = num_pi * sym.pi - (sym.Abs(x) - int(sym.Abs(x) / (num_pi * sym.pi)) * num_pi * sym.pi)
            else:
                ret = x - int(x / (num_pi * sym.pi)) * num_pi * sym.pi
            if ret == 0 and upper:
                ret = sym.sympify(num_pi * sym.pi)
            
        else:  # type(x) is float:
            if x < 0:
                ret = num_pi * np.pi - (np.abs(x) - int(np.abs(x) / (num_pi * np.pi)) * num_pi * np.pi)
            else:
                ret = x - int(x / (num_pi * np.pi)) * num_pi * np.pi
            if ret == 0 and upper:
                ret = sym.sympify(num_pi * np.pi)
                
        return ret

def r2d(t, rad=False, latex=False):
        """
        Converts radians to degrees nicely for use in plots, explanatons, etc. This 
        can take floats (numpy), symbolic (sympy), or string. The output will have the
        same tye as the input.
        
        Usage: 
        
        Note: Initial value must be in radians.
        
        function(data, rad = True/False):
            ...
            r2d('pi/2', rad)
            
        
        """
        if type(t) is str:
            t_ = sym.sympify(t)
            return str(r2d(t_, rad=rad, latex=latex))
        if not rad:
            if str(type(t)).find('sympy') != -1:
                res = sym.sympify(180 / sym.pi * +t)
            else:
                res = 180 / np.pi * t
        else:
            res = t
            
        if latex:
            return sym.latex(res)
        return res
    
def parse_name(nm, G = None, A = None, B = None, N = None, F = None, T = None):
    """
    This will assume that the f_name is given in the form "a + b * f(n * theta)",
    in particular the "n * theta" order is important
    
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
        return parse_name(nm, G, A = A, B = B, N = N, F = F, T = None)
            
    elif B == None:
        if str(type(G)).find('core.mul') != -1:
            if len(G.args) > 2:
                return error()
            B, G = G.args
        else:
            B = 1
        return parse_name(nm, G, A = A, B = B, N = N, F = F, T = None)
    
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
        return parse_name(nm, G, A = A, B = B, N = N, F = F, T = T)
    
    else:
        exp = sym.sympify(A + B*F(N * T))
    
        print(exp)
        print(nm)
    
        if exp == nm:
            return (A, B, N, F, T)
        else:
            return error()
    
        return (A, B, N, F, T)
    


def make_table(col_headers = None, row_headers = None, *entries ):
    """
    This should make a well formatted html table.
    
    Parameters:
    ----------
    col_headers : (Required) list or None
        This should be a list of headers that go across the top of the table.
    row_headers : (Required) list or None
        This should be a list of headers for rows of the table. 
    entries : (Required) list[list], np.array, or var args style (list, list, list, ....) 
        This should fill out the main body of the table and can be alist of rows,
        an array or of the form "list1, list2, ...." each a row. 
    
    Entries could be lists
    """
    
    
    def build_rows(data, row_headers = None, col_headers = None):
        """
        data: list[list]
            It is assumed that the data is in this format
        row_header: list
            If non-empty, it is assumed that len(row_header) = len(data)
        """
        out_string = ""
        
        row_len = len(data[0])
        
        if col_headers is not None:
            
            if row_headers is not None and len(col_headers) == row_len:
                if len(row_headers) == len(data) + 1:
                    col_headers = [row_headers.pop(0)] + col_headers
                elif len(row_headers) == len(data):
                    col_headers = [""] + col_headers
                else:
                    raise ValueError("len(row_headers) should be #rows or #rows + 1")
                
                
            out_string += "<tr class=\'%s\'>" % t_class
            for h in col_headers:
                out_string += "\n<th class=\'%s\'>%s</th>" % (t_class, h)
            out_string += "</tr>"
        
        for j in range(len(data)):
            out_string += "<tr class=\'%s\'>\n" % t_class
            if row_headers is not None:
                out_string += "<th class=\'%s\'>" % t_class + str(row_headers[j]) + "</th>\n" + \
                "".join(["<td class=\'%s\'>" % t_class + \
                         ("%.3f" % i if str(type(i)).find('float') != -1 else "%s" % i) +\
                         "</td>\n" for i in data[j]])
            else:
                out_string += "".join(["<td class=\'%s\'>" % t_class + \
                                       ("%.3f" % i if str(type(i)).find('float') != -1 else "%s" % i) + "</td>\n" 
                                       for i in data[j]])
            out_string += "</tr>\n"
        return out_string

    # do some checks to make sure the entreis are of the correct type
    # entries is a list the items should be one of: (1) lists (all same
    # length) these will be the rows or (2) len(entries) = 1 and entries[0]
    # is a np.array or a list or lists (all same length)

    def validate(entries):

        if len(entries) > 1 and \
            all(map(lambda x: type(x) in [list, type(np.array([]))], entries)) and \
            all(map(lambda x: len(x) == len(entries[0]), entries)):
            data = list(entries)
        elif len(entries) > 1:
            data = [list(entries)]

        if len(entries) == 1:
            if type(entries[0]) is type(np.array([])):
                data = list(entries[0])
                print("First entry is array")
            elif all(map(lambda x: type(x) is list, entries[0])) and \
                        all(map(lambda x: len(x) == len(entries[0][0]), entries[0])):
                data = entries[0]
                print("First entry is list of lists")
            elif type(entries[0]) is list:
                data = list(entries)
                print("First entry is a single list.")
            else:
                raise VaueError("The data is ill formed.")

        return data
    
    
    t_class = "tbl" + str(random.randint(1000, 2000))
    tbl = """
    <style>
        .%s {
            padding: 10px;
            text-align: left;
            border: 0px;
            font-family: sans-serif, serif;
        }
        
        table.%s {
            border-collapse: collapse;
            margin-left: auto;
            margin-right: auto;
        }
        
        div#tblcontainer {
            display: inline-block;
            padding: 5px;
            border-radius: 10px;
            border-style: solid;
            border-width: 1px;
            border-color: rgba(170, 100, 170, .5);
            margin-left: auto;
            margin-right: auto;
        }
        
        div.%s {
            display: block;
            padding: 5px;
            margin-left: 10%%;
            margin-right: 10%%;
            overflow-x: auto;
        }
        
        th.%s {
            font-weight: bold;
            text-align: center;
        }
        
        td.%s {
            text-align: center;
        }
        
        tr.%s:nth-child(odd) {
            background-color: rgba(170, 170, 200, .5);
        }
        
        tr.%s {}
    </style>
    <div class=\'%s\'>
     <div id='tblcontainer'>

    <table class=\'%s\'>
    """ % tuple([t_class] * 9)
    
    data = validate(entries) 
     
    tbl += build_rows(data, row_headers, col_headers)
        
    tbl += "</tr>\n</table>\n</div>\n</div><br>"
    return tbl

if __name__ == "__main__":
    
    f = make_func('Piecewise((sqrt(4 * cos(2 * theta)), 4 * cos(2 * theta) >= 0))', 
              func_params = ('theta'), func_type = 'sympy')
    pts = ['0 + 0', 'pi/4 + 0', 'pi/2 + 0']
    
    print(f(pts))
    print(f([sym.pi/16 * i for i in range(-4,5)]))
    print(f(np.arange(-np.pi/4, np.pi/4,np.pi/16)))
    print(f('pi/8'))
    
            
