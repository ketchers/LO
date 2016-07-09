from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import warnings
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
from chi2TestOfIndependenceData import Chi2TestOfIndependenceData

class Chi2TestOIndependence(object):
    
    def __init__(self, seed = None, path = "chi2toi"):
        self.cache = set() #Just for names this time.
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
            np.random.seed(seed)
        self.hist = ""
        self.solution_plot = ""
        self.path = path
        
        
    def stem(self, context = None, q_type = None, preview = False, 
             force = False):
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
        forrce  : Boolean
            If true force egineration of images.
        
        Notes:
        -----
        The default here is to simulate a roll of a die 30 times and record the
        frequency of values. The :math`\chi^2` test should test whether the die
        is fair at an :math:`alpha` 
        level of 5%.
        """
        
        kwargs = {
            'context': context,
            'q_type': q_type,
            'preview':preview,
            'force':force
            
        }
        
        if q_type is None:
            q_type = random.choice(['STAT', 'HT', 'PVAL'])
          

        if context == None:
            context = Chi2TestOfIndependence()
            
        if not context.is_valid:
            warnings.warn("Context had invalid cell counts.")
            return       
        
        # Generate unique name
        q_name = hash(context)
        self.cache.add(q_name)
        self.hist = str(q_name) + ".png"
        self.solution_plot = str(q_name) + "_sol.png"
        
        style = None
            
        
        if preview:
            question_stem = "<h2>Question</h2><br>"
        else:
            question_stem = ""
        
        question_stem += "<div class='par'>" + context.story + "</div>\n" 
        
        tbl, style = make_table(context.cols, context.rows,
                            True,  context.observed)
        
        question_stem += style + tbl
            
 
        num_rows = len(context.rows)
        num_cols = len(context.cols)
        df =  context.df
        
        chi2eq = "$$\\chi^2_{%s}=\\sum_{i=1}^{%s}\\sum_{j=1}^{%s}\
            \\frac{(O_{i,j}-E_{i,j})^2}{E_{i,j}}\
                = %.3g$$" % (df, num_rows, num_cols, context.chi2_stat)
                
        if q_type == 'STAT':
            question_stem += "Compute the $_\\chi^2$_-statistic and degrees \
                of freedom for the given observations."

        elif q_type == 'PVAL':
           
            question_stem += """The $_\\chi^2$_ statistic is 
                {chi2eq}
                                 
                
            Use this information to find the degrees of freedom (df) and the 
            $_p\\text{{-value}} = P(\\chi^2_{{{df}}} > {chi2:.3g})$_.
            """.format(chi2eq=chi2eq, df = 'df', chi2=context.chi2_stat)
                       
        elif q_type == 'HT':
            
            question_stem += """The degrees of freedom are $_df = 
            ({num_cols} -1)({num_rows} -1) = 
            {df}$_ and the $_\\chi^2$_ statistic is {chi2eq} 
            
            Use this information to conduct a hypothesis test with 
            $_\\alpha = {a_level}$_. Choose the answer that best captures
            the null hypothesis and conclusion.
            """.format(num_cols = num_cols, num_rows = num_rows,  df = df, 
                       chi2eq = chi2eq,
                       a_level = context.a_level)
       
        
        explanation = style
        
        if preview:
            explanation += "<br><h2>Explanation</h2><br>"
                    
        tbl1, _ = make_table(context.cols + ['Total'],
                             context.rows + ['Total'],
                                True,  context.obs_marg) 
           
        tbl2, _ = make_table(context.cols + ['Total'],
                             context.rows + ['Total'],
                                True,  context.expected) 
        
        explanation += """<div class='par'>To find the expected counts multiply 
            the total number of observations by the expected probability 
            for an outcome assuming independence. This is given by
            $$E_{i,j} = \\frac{(\\text{sum of row }i)\\cdot
            (\\text{sum of column }j)}{N}.$$
            The expected counts are shown in the following table:<br><br>
            """
        explanation += tbl2

        explanation += "Recall that the observed counts are:<br><br>"
        explanation += tbl1
        explanation += "</div>"
        
        explanation += """
            <div class='par'>The degrees of freedom are $_df = 
            ({num_cols} - 1)({num_rows} - 1) = \
            {df}$_ and the $_\\chi^2$_-statistic is:{chi2eq}</div>
            """.format(num_cols = num_cols, num_rows = num_rows, 
                   df = df, chi2eq = chi2eq)
        
        if q_type in ['HT','PVAL']:
            
            rv = stats.chi2(df)
            p_val = 1 - rv.cdf(context.chi2_stat)
            
            fname = context.show(path = self.path, force = force)
            
           
            
            img = html_image(fname, width = '300px',  preview = preview)
            
            caption = "Lightshading indicates the p-value.<br>\
                The darker shading indicate the $_\\alpha = %.2g$_ level."\
                % context.a_level
                
            explanation +="""
            The p-value for this data is:
                $$\\text{p-value} = P(\\chi^2_{%s} > %.3g) = %.4g%s$$
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
            """ % (df, context.chi2_stat, p_val * 100, '\\%', img, caption)
            explanation += """
                <div class='par'>The lightly shaded histogram represents a sampling distribution
                of 5000 sample $_\chi^2$_-statistics where the  samples are 
                of the same size as the given observed data (%s observations).
                This indicate how well the $_\chi^2$_-distributions is capturing
                the actual sampling distribution for this experiment.</div>
                """ % context.obs_marg[-1,-1]
        
        
        if q_type == 'HT':
            
            if p_val < context.a_level:
                
                explanation += """
                    <div class='par'>The p-value is less than the $_\\alpha$_-level so
                    the null hypothesis:
                    <br>
                    <strong>H<sub>0</sub>: {null} </strong>
                    </br>
                    is rejected. That is, we accept the alternative hypothesis:
                     <br>
                    <strong>H<sub>a</sub>: {alt} </strong>
                    </br>
                    Precisely, assuming the null hypothesis, there
                    is only a {p_val:.2%} probability due to 
                    random chance in sampling that
                    the difference in the expected and observed data is 
                    least this large.</div>
                    """.format(null=context.null, alt=context.alternative,
                               p_val=p_val)
            else:
                explanation += """
                    <div class='par'>The p-value is greater than the $_\\alpha$_-level so
                    the null hypothesis:
                    <br>
                    <strong>H<sub>0</sub>: {null}</strong>
                    </br>
                    is not rejected. Precisely, assuming the null hypothesis
                    there is a {p_val:.2%} probability due to 
                    random chance in sampling that
                    the difference in the expected and observed data is at 
                    least this large.</div>
                    """.format(null=context.null, p_val=p_val)
                    
            explanation += """
                <div class='par'>Note: {note}</div>                
                """.format(note=context.note)
        
        
#        errors = self.gen_errors(q_type, context)
        
#        if preview:
#            errs = [[er] for er in errors]
#            choices = "<br><h2>Choices</h2><br>"
#            tbl, _ = make_table(None, ['Answer'] + ['Distractor']*4, True, *errs)  
#            choices += tbl                             
            
        return question_stem  +  explanation 
        
 
        
#    def gen_errors(self, q_type, context):
#        N = len(context.outcomes)
#        df = N - 1
#        # A few potential errors
#        # df = N instead of N - 1
#        # use |O_i  - E_i|/E_i instead of (O_i - E_i)^2
#        # use |O_i  - E_i|/E_i instead of (O_i - E_i)^2 and df = N
#        # use (O_i - E_i)/O_i
#        # use (O_i - E_i)/O_i and df = N
#        # take sqrt of chi^2-stat
#           
#        errors = [(N, context.chi2_stat),
#                  (df, np.sum(np.abs(context.t_counts - 
#                      context.o_counts) / context.t_counts)),
#                  (N, np.sum(np.abs(context.t_counts - 
#                      context.o_counts) / context.t_counts)),
#                  (df, np.sum((context.t_counts - 
#                       context.o_counts)**2 / context.o_counts)),
#                  (N, np.sum((context.t_counts - 
#                      context.o_counts)**2 / context.o_counts)),
#                  (df, context.chi2_stat ** .5)]
#                  
#        if q_type == 'STAT':
#        
#            def error_string0(df, chi2):
#                ans = '(degrees of freedom) df = %s and the $_\\chi^2$_-test statistic = %.3g'\
#                       % (df, chi2)
#                return ans
#                
#            ans = error_string0(df, context.chi2_stat)
#            errors = map(lambda x: error_string0(*x), errors)
#            
#            
#            
#            
#        if q_type in ['PVAL']:
#            
#            def error_string1(df, chi2):
#                rv = stats.chi2(df)
#                p_val = 1 - rv.cdf(chi2)
#                
#                ans = '(degrees of freedom) df = %s and the p-value = %.3g'\
#                       % (df, 1 - p_val)
#                return ans
#            
#            ans = error_string1(context.df, context.chi2_stat)
#            errors = map(lambda x: error_string1(*x), errors)
#           
#        if q_type == 'HT':
#            
#            def error_string2(a_level, df, chi2):            
#                rv = stats.chi2(df)
#                p_val = 1 - rv.cdf(chi2)
#                
#                if p_val < a_level:
#                    ans = """
#                        The p-value is {p_val:.2%} and this is less than the 
#                        $_\\alpha$_-level of {a_level}. Therefore we reject the
#                        null hypothesis and find evidence in favor of the 
#                        alternative hypothesis:<br>
#                        <strong>&nbsp;H<sub>1</sub>: {alt}</strong>
#                        """.format(p_val = p_val, a_level = a_level, 
#                                   alt = context.alternative)
#                else:
#                     ans = """
#                        The p-value is {p_val:.2%} and this is greater than the 
#                        $_\\alpha$_-level of {a_level}. Therefore we fail to 
#                        reject the null hypothesis thus supporting the 
#                        hypothesis:<br>
#                        <strong>&nbsp;H<sub>0</sub>: {null}</strong>
#                        """.format(p_val = p_val, a_level = a_level, 
#                                   null = context.null)
#                
#                return ans
#                               
#            ans = error_string2(context.a_level, context.df, context.chi2_stat)
#            errors = map(lambda x: error_string2(context.a_level, *x), errors)    
#        
#        random.shuffle(errors)
#        errors = [ans] + errors[0:4]
#        
#        return errors
    

        
    
        
if __name__ == "__main__":
    
    preview = True
      
    def gen_ctx():
        
        
        ctx = Chi2TestOfIndependenceData()        
        
        
        story = """
            An online survey company puts out a poll asking people two questions. 
            First, it asks if they buy physical CDs. Second, it asks whether they 
            own a smartphone. The company wants to determine if buying physical 
            CDs depends on owning a smartphone.
            """
        
        cd_phone1 = [.1, .9]
        cd_phone2 = [.3, .7]
        cd_no_phone1 = [.4, .6]
        cd_no_phone2 = [.5,.5]
        
        s_sizes = [random.randint(40, 100), random.randint(10, 50)]
        
        rows = ['Smartphone', 'No smartphone']
        cols = ['CD', 'No CD']

        row_dists = [random.choice([cd_phone1, cd_phone2]), random.choice([cd_no_phone1, cd_no_phone2])]        
        
        ctx_phone_cd = Chi2TestOfIndependenceData(story = story, 
                            rows = rows, 
                            cols = cols, 
                            s_sizes = s_sizes, 
                            row_dists = row_dists)
            
        return [ctx, ctx_phone_cd]

    
    prob = Chi2TestOIndependence()
    
    pb = ""
    for q_type in ['STAT','PVAL','HT']:
        for c in gen_ctx():
            result = prob.stem(context = c, preview=preview, q_type = q_type)
            if result is not None:
                pb += '<div class = \'posts\'>'
                pb += result
                pb += '</div><br>'
    print(pb)