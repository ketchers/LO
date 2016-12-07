# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:54:28 2016

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
import pylab as plt



class ArcLengthProb(object):
    
    
    def __init__(self, path = ".", a_type = None, preview = False, 
                 force = False, explanation_type = 'simple'):
        """
        This generate one of the figures for the arc length explanation.

        Parameter:
        ----------
        (kw) path: str
            The path for the image file.
        (kw) explanation_type: str
            Either 'simple' or 'expanded'. If Expanded, then explain where the 
            formula for ds comes from.
        """
        self.path = path
        self.a_type = a_type
        self.force = force
        self.explanation_type = explanation_type
        self.preview = preview
    
    def make_explanation_fig1(self):
        
        
        fname = self.path.rstrip("/") + "/" + "explanation_fig1.png"

        if os.path.isfile(fname) and not self.force:
            print(fname + " exists. Use \"force = True\" or remove file to regenerate it.", 
                 file=sys.stderr)
            return fname

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.axis("off")
        xv = np.linspace(.75,1.75,25)
        yv = xv**2
        xv1 = xv[(1 <= xv) * (xv <= 1.5)]
        yv1 = xv1**2
        ax1.plot(xv,yv,":b")
        ax1.plot(xv1,yv1,"b", lw=4)
        ax1.plot([1,1.5,1.5],[1,1.5**2,1], 'ko')
        ax1.plot([1,1.5],[1,1.5**2], 'k')
        ax1.plot([1,1.5],[1,1], ':k')
        ax1.plot([1.5,1.5],[1,1.5**2], ':k')
        ax1.text(1.2, 1.25**2 + .2, r'$ds$', va = 'bottom', fontsize = 16)
        ax1.text(1.55, 1.2**2 , r'$dy$', ha = 'left', fontsize = 16)
        ax1.text(1.25, .9 , r'$dx$', va = 'top', fontsize = 16)
        ax1.text(1.6, 1.6**2 + .2, r'$C$', va='bottom', fontsize = 16)
        ax1.text(.9, 1.5**2, r'$ds^2 = dx^2 + dy^2$', fontsize =16)
        #ax1.text(1.25, .3, 'Fig 1', fontsize = 16)
        plt.savefig(fname)
        plt.close(fig1)
        return fname
    
    def explain(self):
        
        
        if self.explanation_type == 'simple':
            explanation = ""
        else:
            explanation = """
            The arc length $_s$_ of a curve $_C$_ is given by 
            $$s = \int\,ds,$$ where $_ds$_ is thought of as a tiny piece arc 
            length which satisfies $_ds^2=dx^2+dy^2.$_
            <center>%s</center>
            This can be rearranged as 
            $$ds = \\sqrt{1 + \\left(\\frac{dy}{dx}\\right)^2} \\,dx \quad 
                \\text{ so that }
                \\quad s = \\int_a^b \\sqrt{1 + \\left(\\frac{dy}{dx}\\right)^2}\\,dx$$.
            """%(html_image(self.make_explanation_fig1(), height='auto', width=300,
                         title='Figure indicating why ds^2 = dx^2 + dy^2.',
                         preview=self.preview))
            
        return explanation
        
if __name__ == "__main__":
    e = ArcLengthProb(preview = False, explanation_type = "")
    print(e.explain())