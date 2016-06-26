from __future__ import division
from __future__ import print_function
import sys
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

sys.path.append('finished')
x, y, z = sym.symbols('x,y,z')


class Hyperbola(object):
    """
    Generate a hyperbola of form (x-h)^2 / a^2 - (y - k)^2 / b^2 = 1 if
    trans = 'x' else (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1.

    Parameters:
    ----------

    a,b: numeric or lst[numeric]
        2a is the length of the transverse axis and 2b is the length of the
        conjugate axis.
    h,k: numeric or list[numeric]
        (h,k) is the center of the hyperbola.
    c: numeric
        c is the focal length.
    trans: 'x' (or 'y')
        Which axis is the transversal (major axis) parallel to?
    """
    
    cache = set()
    
    def __init__(self, 
                a = range(1,5), 
                b = range(1,5),
                c = None,
                h = range(-5,5)+[0,0,0], 
                k = range(-5,5) + [0,0,0], 
                trans = ['x', 'y'], latex = False, include_one = False):

    
  
        if type(trans) is list:
            trans = random.choice(trans)
            
        # Pick elements from anything given as a list.
        if type(a) is list:
            a = random.choice(a)
        if type(b) is list:
            b = random.choice(b)
        if type(h) is list:
            h = random.choice(h)
        if type(k) is list:
            k = random.choice(k)
        if type(trans) is list:
            trans = random.choice(trans)
            
        if a is None:
            if type(c) is list:
                c = random.choice(c)
            # It is assumed c is numeric.
            a = sym.sqrt(c**2 - b**2)
            
        if b is None:
            if type(c) is list:
                c = random.choice(c)
            # It is assumed c is numeric.
            b = sym.sqrt(c**2 - a**2)
            
        if c is None:
            c  = sym.sqrt(a**2 + b**2)
        
         
            
        # At this point, if preconditions are met a, b, and c are defined. 
        self.a = a
        self.b = b
        self.c = c
        self.h = h
        self.k = k
        self.trans = trans
    
       
        if trans == 'x':
            x_expr =  sym.sympify((x - h)**2 / a**2)
            y_expr =  sym.sympify((y - k)**2 / b**2)
            expr = x_expr - y_expr
        else:
            x_expr =  sym.sympify((x - h)**2 / b**2)
            y_expr =  sym.sympify((y - k)**2 / a**2)
            expr = y_expr - x_expr

        self.x_expr = x_expr
        self.y_expr = y_expr
        self.expr = expr
        self.name = str(expr)
        self.url = quote_plus(self.name)
        
        if trans == 'x':
            self.latex = sym.latex(x_expr, long_frac_ratio=sym.oo) + " - " + \
              sym.latex(y_expr, long_frac_ratio=sym.oo)
            self.asymptote1 = sym.sympify(b / a * (x - h) + k)
            self.asymptote2 = sym.sympify(-b / a * (x - h) + k)
        else:
            self.latex = sym.latex(y_expr, long_frac_ratio=sym.oo) + " - " + \
              sym.latex(x_expr, long_frac_ratio=sym.oo)
            self.asymptote1 = sym.sympify(a / b * (x - h) + k)
            self.asymptote2 = sym.sympify(-a / b * (x - h) + k)

        self.hash = 17

        if self not in Hyperbola.cache:
            Hyperbola.cache.add(self)

    def __hash__(self):
        if self.hash == 17:
            for field in [self.a, self.b, self.h, self.k, self.trans]:
                self.hash += self.hash + 31 * self.hash
        return self.hash

    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self.a == other.a and self.b == other.b and self.h == other.h \
            and self.k == other.k and self.trans == other.trans

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.name

    def show(self, path=".", file_name=None, force=False, \
        img_type="png", xkcd = False, label = False):
        """
        This creates a plot of the hyperbola, if it does not already exist.

        Parameters:
        ----------
        file_name     : str
            Name  of image. If 'show', then just display.
        path          : str
            Path to store image. The folder is created if does not exist
        force         : Boolean
            Force rbuild of images if they exist.
        preview       : Boolean
            Format output for preview
        img_type      : str
            The type of image: "png", "gif", ...
        """


        if file_name is None:
            file_name_ = self.url
        else:
            file_name_ = file_name
    
        fname = path + "/" + file_name_ + "." + img_type
        if os.path.isfile(fname) and not force and file_name != 'show':
            print("The file \'" + fname + "\' exists, \
            not regenerating. Delete file to force regeneration.", file=sys.stderr)
            return fname.replace('%2', '%252')

        d = 2 * int(float(self.c))
        xmin = self.h - d
        xmax = self.h + d
        ymin = self.k - d
        ymax = self.k + d


        X = np.linspace(xmin, xmax, 200)
        Y = np.linspace(ymin, ymax, 200)
        X,Y = plt.meshgrid(X,Y)
        f = make_func(self.expr, ('x', 'y'))
        Z = f(X,Y)
        
        
        # The plotting code
        if xkcd:
            plt.xkcd()        
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for label in ['bottom', 'left']:
            ax.spines[label].set_position('zero')  # this is what zeros the axes
            ax.spines[label].set_linewidth(3)
            ax.spines[label].set_alpha(0.6)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        ax.set_xticks(tools.non_zero_range(xmin + 1, xmax))
        ax.set_yticks(tools.non_zero_range(ymin + 1, ymax))
        # Plot the hyperbola
        c_plot = plt.contour(X,Y,Z,[1])
                
                   
                
        # Plot the central rectangle
        if self.trans == 'x':
            a = self.a
            b = self.b
        else:
            a = self.b
            b = self.a

        plt.plot([self.h + a, self.h - a, self.h - a, \
                  self.h + a, self.h + a],
                 [self.k + b, self.k + b, self.k - b, \
                  self.k - b, self.k + b], color = (.3, .7, .3, .5),
                 ls = "dashed", lw = 2)

        plt.plot([self.h + a, self.h - a, self.h, self.h], 
                 [self.k, self.k, self.k + b, self.k - b],
                 'o', color = (1,0,0,.5), lw=20)
        if self.trans == 'x':
            plt.plot([self.h - self.c, self.h + self.c], 
                     [self.k, self.k], 'o', color = (0, 0, 1, .5), lw = 20)
        else:
            plt.plot([self.h, self.h], 
                     [self.k  - self.c, self.k  + self.c], 
                     'o', color = (0, 0, 1, .5), lw = 20)

        plt.plot([xmin, xmax], 
                 [b/a*(xmin - self.h) + self.k, b/a*(xmax - self.h) + self.k],
                color = (1,0,1,.5), ls = 'dashed')
        plt.plot([xmin, xmax], 
                 [-b/a*(xmin - self.h) + self.k, -b/a*(xmax - self.h) + self.k],
                color = (1,0,1,.5), ls = 'dashed')
        
        plt.grid()
        
        
        if file_name == 'show':
            plt.show()
            plt.close()
        else:
            tools.make_folder_if_necessary(".", path)
            plt.savefig(fname) 
            plt.close()
            
        return fname.replace('%2','%252')
            
            
    def explanation(self, path = 'explanations', preview = False,                     
                    force = False, xkcd = False):
        """
        Provides an explanation of the hyperbola.
        
        Parameters: (Largely shared with show)
        -----------
        path      : str
            The output directory for image files
        file_name : str
            The name of output file without extension.
        preview   : Boolean
            If true thing are set up for peviewing. (Images, ...)
        """
        ex = ""
        
       
        file_name = path + "/" + self.url + ".png"
        
        if preview:
            file_name = file_name.replace('%2','%252')
         
        ex += "The graph of the hyperbola $$%s = 1$$ is:" % self.latex
        
        if self.trans == 'x':
            tbl, style = make_table(None, None, False, 
                [
                    ['center', '$_(%s, %s)$_' % (self.h, self.k)],
                    ['vertices', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.h - self.a, self.k, self.h 
                        + self.a, self.k]))],
                    ['length of conjugate axis', '$_%s$_' % sym.latex(2*self.a)],
                    ['co-vertices', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.k - self.b, self.h, self.k 
                        + self.b, self.h]))],
                    ['length of conjugate axis', '$_%s$_'% sym.latex(2*self.b)],
                    ['foci', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.h - self.c, self.k, self.h 
                        + self.c, self.k]))],
                    ['asymptotes', '$_y = \\pm\\frac{%s}{%s}(%s) %+d$_' \
                        % tuple(map(sym.latex, [self.b, self.a, 
                                                sym.sympify(x - self.h)]) + 
                                                [int(self.k)])]             
                ])
        else:
            tbl, style = make_table(None, None, False,
                [
                    ['center', '$_(%s, %s)$_' % (self.h, self.k)],
                    ['vertices', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.h, self.k - self.a, self.h, 
                                                self.k + self.a]))],
                    ['length of conjugate axis', '$_%s$_' % sym.latex(2*self.a)],
                    ['co-vertices', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.k, self.h - self.b, self.k, 
                                                self.h + self.b]))],
                    ['length of conjugate axis', '$_%s$_'% sym.latex(2*self.b)],
                    ['foci', '$_(%s, %s), (%s, %s)$_' \
                        % tuple(map(sym.latex, [self.h, self.k - self.c, self.h, 
                                                self.k + self.c]))],
                    ['asymptotes', '$_y = \\pm\\frac{%s}{%s}(%s) %+d$_' \
                        % tuple(map(sym.latex, 
                                    [self.a, self.b, sym.sympify(x - self.h)]) + 
                                    [int(self.k)])]             
                ])
                
        img = html_image(image_url = file_name, width = '300px', preview = preview)
        
        ex +="""      
        %s
        <div class='outer-container-rk'>
            <div class='centering-rk'>
                <div class='container-rk'>
                    <figure>
                        %s
                        <figcaption>$_%s = 1$_</figcaption>
                    </figure>
                </div>
                
                <div class='container-rk'>
                    %s
                </div>
            </div>
        </div>
        """ % (style, img, self.latex, tbl)
            
        
        self.show(path = path, file_name = self.url, force = force, 
                  xkcd = xkcd)
        
        
        return ex
        
  
if __name__ == "__main__":
    path = 'hyperbola'
    h = Hyperbola(a=3,b=4,trans='x')
    print(h.show(path=path, label=True))      
    print(h.explanation(path=path, preview=True))
 