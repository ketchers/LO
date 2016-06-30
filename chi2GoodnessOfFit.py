from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import tools
import random
import sympy as sym
import numpy as np
import matplotlib as mpl
import scipy as sp
import scipy.stats as stats
import pylab as plt
plt.rc('font', family='DejaVu Sans')
x, y, z = sym.symbols('x,y,z')
from html_tools import make_table, html_image
from chi2GoodnessOfFitData import Chi2GoodnessOfFitData

class Chi2GoodnessOfFit():
    
    def __init__(self, seed = None, path = "chi2gof"):
        self.cache = set() #Just for names this time.
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
        self.hist = ""
        self.solution_plot = ""
        self.path = path
        
        
    def stem(self, context = None, table = None, q_type = 'statistic',
             preview = False):
        """
        This will generate a problem for $\chi^2$ goodness of fit.
        
        Parameters:
        ----------
        context : context object
            This describes the problem. A default context is used if this is 
            none.
        table   : string ['hist', 'table'] 
            Display t_dist as a table or as a histogram.
        q_type  : string [None, 'STAT', 'HT', 'CI'] 
            If None randomly choose. If 'STAT' just compute the chi2 statistic 
            and the degrees of freedom. If 'HT, compute the p-value for the  
            data and determine whether or not reject the null hypothesis. If 
            'CI' compute the confidence interval.
        
        Notes:
        -----
        The default here is to simulate a roll of a die 30 times and record the
        frequency of values. The :math`\chi^2` test should test whether the die
        is fair at an :math:`alpha` 
        level of 5%.
        """
        
        kwargs = {
            'context': context,
            'table': table,
            'q_type': q_type,
            
        }
        
        if q_type is None:
            q_type = random.choice(['STAT', 'HT', 'PVAL'])
          

        if table == None:
            table = random.choice(['table', 'hist'])
        
        if context == None:
            context = Chi2GoodnessOfFitData()
            
        # Generate unique name
        q_name = random.randint(1, 20000)
        while q_name in self.cache:
            q_name = random.randint(1, 20,000)
        self.cache.add(q_name)
        self.hist = self.path + "/" + str(q_name) + ".png"
        self.solution_plot = self.path + "/" + str(q_name) + "_sol.png"
            
        
        question_stem = "<div>" + context.story + "</div>\n" 
        if table == 'table':
            html, style = make_table(context.outcomes,
                                [context.outcome_type, 
                                'Observed Counts'],
                                True,  context.o_counts)
            question_stem += style + html
            
        elif table == 'hist':
            self.gen_observed_hist(context.outcomes, context.t_dist, 
                              context.s_size, context.sample)
                                  
            img = html_image(self.hist, width = '300px', preview = preview)
            
            
        
            _, style = make_table(None,None,True,[1])
            
            question_stem +="""      
            %s
            <div class='outer-container-rk'>
                <div class='centering-rk'>
                    <div class='container-rk'>
                        <figure>
                            %s
                            <figcaption>%s</figcaption>
                        </figure>
                    </div>
                </div>
            </div>
            """ % (style, img, "Observed Frequencies")                
        
        if q_type == 'STAT':
            question_stem += "Compute the $_\\chi^2$_-statistic and degrees \
                of freedom for the given observed values."
        elif q_type == 'HT':
            df = len(context.outcomes) - 1
            chi2eq = "$$\\chi^2_{%s}=\\sum_{i=1}^{%s}\\frac{(O_i-E_i)^2}{E_i}\
                = %.3g.$$" % (df, len(context.outcomes), context.chi2_stat)
            
            question_stem += """The degrees of freedom are $_df = $_ {df} and 
            the $_\\chi^2$_ statistic is {chi2} 
            
            Use this information to conduct a hypothesis test with 
            $_\\alpha = {a_level}$_. Choose the answer that best captures
            the null hypothesis and conclusion.
            """.format(df = len(context.outcomes) - 1,
                       chi2 = chi2eq,
                       a_level = context.a_level)
        elif q_type == 'PVAL':
            df = len(context.outcomes) - 1
            chi2eq = "$$\\chi^2_{df}=\\sum_{i=1}^{%s}\\frac{(O_i-E_i)^2}{E_i}\
                = %.3g.$$" % (df, context.chi2_stat)
            
            question_stem += """The $_\\chi^2$_ statistic is {chi2} 
            
            Use this information to find the $_p$_-value and 
            degrees of freedom.
            """.format(df = len(context.outcomes) - 1,
                       chi2 = chi2eq)
                            
        explanation = "<br><br><strong>The explanation is in progress not done,\
        but here is part of what they get:</strong>"
        
        img = html_image(self.solution_plot, width = '300px', 
                             preview = preview)
            
        tbl, style = make_table(context.outcomes,
                                [context.outcome_type, 'Expected Counts', 
                                'Observed Counts'],
                                True,  context.t_counts, context.o_counts)
            
        self.gen_solution_plot(context.chi2_stat, context.outcomes, 
                              context.t_dist, context.s_size, 
                              a_level = context.a_level)
                              
            
        
        explanation +="""    
        %s
        %s
        <div class='outer-container-rk'>
            <div class='centering-rk'>
                <div class='container-rk'>
                    <figure>
                        %s
                        <figcaption>%s</figcaption>
                    </figure>
                </div>
            </div>
        </div>
        """ % (style, tbl, img, "Observed Frequencies")         
        
        return question_stem + "<div>" + explanation +"</div>"
        
        
    def gen_solution_plot(self, chi2_stat, outcomes, t_dist, s_size, a_level = 0.05):

        rv = stats.chi2(len(t_dist) - 1)
        x_data = np.linspace(max([0,rv.mean()- 3*rv.std()]), rv.mean( )+ 4*rv.std(), 200)
        y_data = rv.pdf(x_data)
        t = rv.ppf(1 - a_level)
        x_above_crit = x_data > t
        x_above_statistic = x_data > chi2_stat
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_data,y_data)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('bottom')
        ax.text((min(x_data) + max(x_data)) / 2 , (min(y_data) + max(y_data)) / 2,
                "p-value = %.3f%s" % ((1-rv.cdf(chi2_stat)) * 100, '%'), fontsize = 14)
        
        ax.plot([chi2_stat,chi2_stat], [0,max(.005,rv.pdf(chi2_stat))], 'r-', lw = 2)
        ax.fill_between(x_data[x_above_crit ], 0, y_data[x_above_crit], color = (.7,.2,.7,.5))
        ax.fill_between(x_data[x_above_statistic], 0, y_data[x_above_statistic], color = (.7,.2,.7,.3))
    
    
        s = np.random.choice(outcomes, (5000,s_size), p = t_dist)
        q = tuple(np.transpose([np.sum(s == i, axis = 1)]) for i in outcomes)
        q = np.sum((np.hstack(q) - 
                    s_size*np.ones((5000, len(t_dist)))*np.array(t_dist)) **2 / (s_size * np.array(t_dist))
                   , axis=1)
        q = q[q < rv.mean( )+ 4*rv.std()]
        ax.hist(q, bins=15, normed=True, color = (.8,.8,1,.2), histtype='stepfilled', lw=1, ls=":")
        tools.make_folder_if_necessary(".", self.path)        
        plt.savefig(self.solution_plot) 
        plt.close()
    
        
    def gen_observed_hist(self, outcomes, dist, s_size, observed = None, 
                          rotation = None):

        outcome_dict = dict([(b, a) for a,b in enumerate(outcomes)])
        if observed is None:
            observed = np.random.choice(range(len(outcomes)), s_size, p = dist)
        else:
            observed = [outcome_dict[a] for a in observed]
    
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks(range(len(dist)))
        
        if rotation is None:
            try:
                if any(map(lambda x: len(x) > 7, outcomes)):
                    rotation = -60
                elif any(map(lambda x: len(x) > 4, outcomes)):
                    rotation = -30
                else:
                    rotation = 0
            except:
                rotation = 0
        
        ax.set_xticklabels(outcomes, rotation = rotation, size=16)
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('bottom')
       
    
        freq = ax.hist(observed, bins = np.arange(-.5,len(dist) + .5, 1), 
                       color = (.7,.3,.7,.5))[0]
        #ax.set_title("Observed Frequencies", y = .1)
        [ax.text(i, freq[i], "%0.4g" % freq[i],ha = 'center', va = 'bottom') 
            for i in range(len(dist))] 

        tools.make_folder_if_necessary(".", self.path)        
        plt.savefig(self.hist)    
        plt.close()
        
if __name__ == "__main__":
    
    preview = True
      
    def gen_ctx():
        ctx = Chi2GoodnessOfFitData()
    
        # Use unicode dice
        ctx1 = Chi2GoodnessOfFitData(outcomes = \
               [u'\u2680', u'\u2681', u'\u2682', u'\u2683', u'\u2684', u'\u2685'])
                
        
        # Pass the pigs game
        outcomes =  ['Pink', 'Dot', 'Razorback', 'Trotter', 'Snouter', 'Leaning Jowler']
        t_dist = [.35, .30, .20, .10, .04, .01]
        tbl, styles = make_table(outcomes, ['Position', 'Expected Frequency'],True, 
                                t_dist)
        
        story = """Pass The Pigs&reg; is a game from Milton-Bradley&#8482; which is 
                essentially a dice game except that instead of dice players toss
                small plastic pigs that can land in any of 6 positions. For example, 
                you roll a trotter if the pig falls standind on all 4 legs. 
                The expected for the 6 positions are:
                
                {styles}
                {tbl}            
                """.format(styles = styles, tbl = tbl)
                
        ctx2 = Chi2GoodnessOfFitData(
            outcomes = outcomes,
            t_dist = t_dist,
            s_size = random.randint(5, 20) * 6,
            a_level = random.choice([0.1,0.01,0.05]),
            story = story)
        
        return [ctx, ctx2]
            
    
    
    prob = Chi2GoodnessOfFit()
    pb = ""
    for q_type in ['STAT', 'HT', 'PVAL']:
        for table in ['hist', 'table']:
            for context in ['ctx', 'ctx1']:
                c = random.choice(gen_ctx())
                pb += '<div class = \'posts\'>'
                pb += prob.stem(context = c, preview=preview, q_type = q_type)
                pb += '</div><br>'
    print(pb)