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
        
    def __call__(self, *input, **kwargs):

        if 'func_type' not in kwargs:
            func_type = None
        else:
            func_type = kwargs['func_type']
        
        if func_type is None:
            is_np = False
            for i in input:
                if type(i) is np.ndarray and i.dtype.type is np.float_:
                    is_np = is_np | True
            if is_np:
                func_type = 'numpy'
            else:
                func_type = 'sympy'        
        
        f_ = make_func(self.expr, func_params=('x','y'), func_type=func_type)
        
        return f_(*input)


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
            
            
    def explanation(self, path = 'explanations', 
                expanded = True, preview = False,                     
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
    
    
        const = self.a ** 2 * self.b **2
        x_part = const*(self(x, self.k).expand() - self(0, self.k))
        y_part = const*(self(self.h, y).expand() - self(self.h, 0))
        rhs = const - const*self(0, 0)
    
        # The following is an attempt to avoid lots of code duplication.
        if self.trans == 'x':
            X = x
            Y = y
            X_part = x_part
            Y_part = y_part
            HH = self.h
            KK = self.k
            TX = 1
            TY = 0
        else:
            X = y
            Y = x
            X_part = y_part
            Y_part = x_part
            HH = self.k
            KK = self.h
            TX = 0
            TY = 1
        # To reduce typing
        A = self.a
        B = self.b
        H = self.h
        K = self.k
        C = self.c 
    
        k, h, a, b, c = sym.symbols('k h a b c')
    
        gcd_X = sym.gcd(X_part.collect(X).coeff(X,1), X_part.collect(X).coeff(X,2))
        gcd_Y = sym.gcd(Y_part.collect(Y).coeff(Y,1), Y_part.collect(Y).coeff(Y,2))
        b_X = X_part.collect(X).coeff(X,1)/gcd_X
        b_Y = -Y_part.collect(Y).coeff(Y,1)/gcd_Y        
    
        if expanded:
            ex += "The first step to finding the graph of $$%s = 0$$ is\
                to find the normal form of the equation." \
            % sym.latex((const*(self(x, y) - 1)).expand())
    
            ex += " To begin move the constant term to the right hand side. \
                This gives: $$(%s) + (%s) = %s$$" \
                    % (sym.latex(x_part), sym.latex(y_part), rhs)
    
    
    
            ex += "Next factor common factors from the $_x$_ and $_y$_ terms to get: \
                $$%s(%s) - %s(%s) = %s$$" \
            % (gcd_X if gcd_X != 1 else "", sym.latex(X_part/gcd_X), 
               gcd_Y if gcd_Y != 1 else "", sym.latex(-Y_part/gcd_Y), rhs)
    
            ex += "Now complete the squares: \
                $$%s\\left(%s  %s\\right) \
                - %s\\left(%s  %s\\right) = \
                %s %s %s$$"\
                 % (gcd_X if gcd_X != 1 else "", sym.latex(X_part/gcd_X), 
                    "+ \\left(\\frac{%s}{2}\\right)^2" % b_X if b_X != 0 else "", 
                    gcd_Y if gcd_Y != 1 else "", sym.latex(-Y_part/gcd_Y), 
                    "+ \\left(\\frac{%s}{2}\\right)^2" % b_Y if b_Y != 0 else "", 
                    rhs, 
                    "+ %s\\left(\\frac{%s}{2}\\right)^2" % (gcd_X, b_X) if b_X != 0 else "", 
                    "- %s\\left(\\frac{%s}{2}\\right)^2" % (gcd_Y, b_Y) if b_Y != 0 else "")
    
            ex += "This simplifies to: $$%s\\left(%s\\right)^2 - %s\\left(%s\\right)^2 = %s$$" \
                % (gcd_X if gcd_X != 1 else "",
                   sym.latex(X + b_X/2),
                   gcd_Y if gcd_Y != 1 else "",
                    sym.latex(Y + b_Y/2),
                   const)
    
            ex += "Lastly divide both side by the right hand side to get: \
                $$%s\\left(%s\\right)^2 - %s\\left(%s\\right)^2 = 1$$" \
                % (sym.latex(gcd_X/const) if gcd_X/const != 1 else "",
                   sym.latex(X + b_X/2),
                   sym.latex(gcd_Y/const) if gcd_Y/const != 1 else "",
                    sym.latex(Y + b_Y/2))
    
            ex += "This simplifies to the final normal form: \
                $$\\frac{(%s)^2}{%s^2} - \\frac{(%s)^2}{%s^2} = 1$$" \
                % (sym.latex(X-HH), A, sym.latex(Y-KK), B)
        else:
            ex += "The hyperbola is given in standard normal form: \
            $$\\frac{(%s)^2}{%s^2} - \\frac{(%s)^2}{%s^2} = 1$$" \
                % (sym.latex(X-HH), A, sym.latex(Y-KK), B)
    
        ex += "From this we can read off the center to be at $_(h,k) = (%s, %s)$_. "\
            % (H,K)
    
        ex += "The tansverse (major) axis is along $_%s = %s$_ and has length $_2a = %s$_."\
            % (sym.latex(Y), K, 2*A)
    
        ex += "The vertices are $_(%s, %s) = (%s,%s)$_ and $_(%s, %s) = (%s, %s)$_."\
            % (sym.latex(h - a * TX), sym.latex(k - a * TY), H - A * TX, K - A * TY,
               sym.latex(h + a * TX), sym.latex(k + a * TY), H + A * TX, K + A * TY)
    
        ex += "The conjugate (minor) axis is along $_%s = %s$_ and has length \
              $_2b = %s$_." % (sym.latex(Y), H, 2*B)
    
        ex += "The co-vertices are $_(%s, %s) = (%s,%s)$_ and $_(%s, %s) = (%s, %s)$_."\
            % (sym.latex(h - B * (1 - TX)), sym.latex(k - B * (1 - TY)), 
                    H - B * (1 - TX), K - B * (1 - TY), 
                    sym.latex(h + B * (1 - TX)), sym.latex(k + B * (1 - TY)), 
                    H + B * (1 - TX), K + B * (1 - TY))
    
        ex += "The two assymptotes are $_y = \\pm\\frac{b}{a}(%s) %s %s \
              = \\pm\\frac{%s}{%s}(%s) %s %s $_. " \
                % (sym.latex(x - H),  "+" if K > 0 else "-",
                    sym.Abs(K), B, A, sym.latex(x - H), "+" if K > 0 else "-",
                    sym.Abs(K))
    
        ex += "Finally, the focal length is $_c = \\sqrt{a^2+b^2}=%s$_ and the foci \
            are located at $_(%s, %s) = (%s,%s)$_ and $_(%s, %s) = (%s, %s)$_."\
            % (sym.latex(C),
                    sym.latex(h - c * TX), sym.latex(k - c * TY), 
                    sym.latex(H - C * TX), sym.latex(K - C * TY), 
                    sym.latex(h + c * TX), sym.latex(k + c * TY), 
                    sym.latex(H + C * TX), sym.latex(K + C * TY))
    
    
        tbl, style = make_table(None, None, False, 
            [
                ['center', '$_(%s, %s)$_' % (H, K)],
                ['vertices', '$_(%s, %s), (%s, %s)$_' \
                    % tuple(map(sym.latex, [H - self.a * TX, K - self.b * TY, 
                                            H + self.a * TX, K + self.b *TY]))],
                ['length of conjugate axis', '$_%s$_' % sym.latex(2*A)],
                ['co-vertices', '$_(%s, %s), (%s, %s)$_' \
                    % tuple(map(sym.latex, [H - self.a * (1 - TX), K - self.b * (1 - TY), 
                                            H + self.a * (1 - TX), K - self.b * (1 - TY)]))],
                ['length of conjugate axis', '$_%s$_'% sym.latex(2*B)],
                ['foci', '$_(%s, %s), (%s, %s)$_' \
                    % tuple(map(sym.latex, [H - C * TX, K - C * TY, 
                                            H + C * TX, K + C * TY]))],
                ['asymptotes', '$_y = \\pm\\frac{%s}{%s}(%s) %s %s$_' \
                    % tuple(map(sym.latex, [B, A, sym.sympify(x - H), 
                                            "+" if K > 0 else "-",
                                            sym.Abs(K)]))] 
    
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
 