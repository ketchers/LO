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
from html_tools import html_image
from table import Table
from chi2TestOfHomogeneityData import Chi2TestOfHomogeneityData

class Chi2TestOfHomogeneity(object):
    
    def __init__(self, seed = None, path = "chi2toh"):
        self.cache = set() #Just for names this time.
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
            np.random.seed(seed)
        self.hist = ""
        self.solution_plot = ""
        self.path = path
        
        
    def stem(self, context = None, q_type = None, a_type = 'preview', 
             force = False):
        """
        This will generate a problem for $\chi^2$ test of homogeneity.
        
        Parameters:
        ----------
        context : Chi2TestOfHomogeneity object
            This describes the problem. A default context is used if this is 
            none.
        q_type  : string [None, 'STAT', 'HT', 'CI'] 
            If None randomly choose. If 'STAT' just compute the chi2 statistic 
            and the degrees of freedom. If 'HT, compute the p-value for the  
            data and determine whether or not reject the null hypothesis. If 
            'CI' compute the confidence interval.
        force  : Boolean
            If true force egineration of images.
        a_type  : string
            This is eithe "MC" or "preview" for now
        """
        
        kwargs = {
            'context': context,
            'q_type': q_type,
            'a_type': a_type,
            'force': force
            
        }
        
        if q_type is None:
            q_type = random.choice(['STAT', 'HT', 'PVAL'])
          

        if context == None:
            context = Chi2TestOfHomogeneityData()
            
        if not context.is_valid:
            warnings.warn("Context had invalid cell counts.")
            return       
        
        # Generate unique name
        q_name = hash(context)
        self.cache.add(q_name)
        self.hist = str(q_name) + ".png"
        self.solution_plot = str(q_name) + "_sol.png"
        
        style = None
            
        
        if a_type == 'preview':
            question_stem = "<h2>Question</h2><br>"
        else:
            question_stem = ""
        
        if fmt == 'html':
            question_stem += Table.get_style()        
        
        question_stem += "<div class='par'>" + context.story + "</div>\n" 
       
        if fmt == 'html':
            tbl = context.observed_table.html()
        else:
            tbl = context.observed_table.latex()
        
        question_stem += "\n" + tbl + "\n"
 
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
       
        
        if fmt == 'html':
            explanation = Table.get_style()
        else:
            explanation = ""
        
        if a_type == 'preview':
            explanation += "<br><h2>Explanation</h2><br>"
                    
        if fmt == 'html':
            tbl1 = context.obs_marg_table.html()
            tbl2 = context.expected_table.html()
        else:
            tbl1 = context.obs_marg_table.latex()
            tbl2 = context.expected_table.latex()
        
        explanation += """<div class='par'>To find the expected counts assuming
            homogeneity of distributions, ffirst find the column totals and
            divide by the size of the sample. This provides the overall 
            distribution. Let $$p_j = \\frac{(\\text{sum of column }i)}{N},$$
            where $_N$_ is the total size of the population.</div>
            
            <div class='par'>To compute the expected count for the 
            $_(i,j)^{\\text{th}}$_&nbsp; cell, multiply the 
            the sum of the observed values in the $_i^{\\text{th}}$_&nbsp; 
            row by $_p_i$_. This gives $_E_{i,j}$_.</div>
            
            <div class='par'>These two steps can be combined to give:
            $$E_{i,j} = \\frac{(\\text{sum of row }i)\\cdot
            (\\text{sum of column }j)}{N}.$$
            Notice that this is exactly the same computation as for a
            $_\\chi^2$_-test of independence.</div>            
            
            <div class='par'>The expected counts are shown in the following 
            table:<br><br>
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
            
           
            
            img = html_image(fname, width = '300px',  
                             preview = (a_type == 'preview'))
            
            caption = """
                Lightshading (right of the red line) indicates the \p-value.
                <br>
                The darker shading indicate the $_\\alpha = $_ {a_level:.0%} 
                level.<br>
                The background histogram is a bootstrap sampling distribution.
                """.format(a_level = context.a_level)
                
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
                    
            if context.note is not None:
                explanation += """
                    <div class='par'>Note: {note}</div>                
                    """.format(note=context.note)
        
        
        errors = self.gen_errors(q_type, context)
                 
      
        if a_type == 'preview':

            errs = [[er] for er in errors]

            choices = "\n<br><h2>Choices</h2><br>\n"

            tb = Table(errs, row_headers = ['Answer'] + ['Distractor']*4)
                       
           
            tbl = tb.html()
            
            choices += Table.get_style() + "\n" + tbl  

            if fmt == 'html':
                return question_stem + choices +  explanation
            else:
                return question_stem.replace("<div ","<p ")\
                        .replace("</div>","</p>") + choices + \
                      explanation.replace("<div ","<p ")\
                        .replace("</div>","</p>")
            
            return question_stem + choices +  explanation 
            
        elif a_type == 'MC':
            
            if fmt == 'latex':
                question_stem = question_stem.replace("<div ","<p ")\
                        .replace("</div>","</p>")
                explanation = explanation.replace("<div ","<p ")\
                        .replace("</div>","</p>")
                distractors = [err.replace("<div ","<p ")\
                        .replace("</div>","</p>") for err in errors]
           
            
            question_stem = ' '.join(question_stem.split())
            distractors = [' '.join(err.split()) for err in errors]
            explanation = ' '.join(explanation.split()) + "\n"
            return tools.fully_formatted_question(question_stem, explanation, 
                                                  answer_choices=distractors)

        elif a_type == 'Match':
            pass
        else:
            pass    
        
 
        
    def gen_errors(self, q_type, context):
        N = len(context.cols)
        M = len(context.rows)
        df = context.df
        observed = context.observed
        expected = context.expected[:-1,:-1]
        T = context.obs_marg[-1,-1]
        # A few potential errors
        # df = M*M instead of (N-1)(M-1)
        # use |O_ij  - E_ij|/E_ij instead of (O_ij - E_ij)^2
        # use |O_i  - E_i|/E_i instead of (O_ii - E_ij)^2 and df = NM
        # use (O_i - E_i)/O_i
        # use (O_i - E_i)/O_i and df = N
        # take sqrt of chi^2-stat
           
        errors = [(N*M, context.chi2_stat),
                  (df, np.sum(np.abs(observed - expected) / expected)),
                  (N*M, np.sum(np.abs(observed - expected) / expected)),
                  (df, np.sum((observed - expected)**2 / observed)),
                  (N*M, np.sum((observed - expected)**2 / observed)),
                  (df, context.chi2_stat ** .5)]
                  
        if q_type == 'STAT':
        
            def error_string0(df, chi2):
                ans = '(degrees of freedom) df = %s and the $_\\chi^2$_-test \
                    statistic = %.3g' % (df, chi2)
                return ans
                
            ans = error_string0(df, context.chi2_stat)
            errors = map(lambda x: error_string0(*x), errors)
            
            
            
            
        if q_type in ['PVAL']:
            
            def error_string1(df, chi2):
                rv = stats.chi2(df)
                p_val = 1 - rv.cdf(chi2)
                
                ans = '(degrees of freedom) df = %s and the p-value = %.3g'\
                       % (df, p_val)
                return ans
            
            ans = error_string1(context.df, context.chi2_stat)
            errors = map(lambda x: error_string1(*x), errors)
           
        if q_type == 'HT':
            
            def error_string2(a_level, correct, df, chi2):            
                rv = stats.chi2(df)
                p_val = 1 - rv.cdf(chi2)
                
                if p_val < a_level and correct:
                    ans = """
                        The p-value is {p_val:.2%} and this is less than the 
                        $_\\alpha$_-level of {a_level}. Therefore we reject the
                        null hypothesis and find evidence in favor of the 
                        alternative hypothesis:<br>
                        <strong>&nbsp;H<sub>1</sub>: {alt}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   alt = context.alternative)
                                   
                elif p_val >= a_level and correct:
                     ans = """
                        The p-value is {p_val:.2%} and this is greater than the 
                        $_\\alpha$_-level of {a_level}. Therefore we fail to 
                        reject the null hypothesis thus supporting the 
                        hypothesis:<br>
                        <strong>&nbsp;H<sub>0</sub>: {null}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   null = context.null)
                                   
                elif p_val < a_level and not correct:
                    ans = """
                        The p-value is {p_val:.2%} and this is less than the 
                        $_\\alpha$_-level of {a_level}. Therefore we fail to 
                        reject the null hypothesis thus supporting the 
                        hypothesis:<br>
                        <strong>&nbsp;H<sub>0</sub>: {null}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   null = context.null)
                else:
                     ans = """
                        The p-value is {p_val:.2%} and this is greater than the 
                        $_\\alpha$_-level of {a_level}. Therefore we reject the
                        null hypothesis and find evidence in favor of the 
                        alternative hypothesis:<br>
                        <strong>&nbsp;H<sub>1</sub>: {alt}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   alt = context.alternative)
                    
                
                return ans
                               
            ans = error_string2(context.a_level, True, context.df, context.chi2_stat)
            errors = map(lambda x: error_string2(context.a_level, 
                            random.choice([True, False]), *x), errors)  
        
        random.shuffle(errors)
        errors = [ans] + errors[0:4]
        
        return errors
    

        
    
        
if __name__ == "__main__":
    
    a_type = 'preview'
    fmt = 'html'
    seed = 44
      
    def gen_ctx(seed = seed):
        
        
        # Default context        
        ctx = Chi2TestOfHomogeneityData(seed = seed)        
        
        # A non default context with a little randomness thrown into
        # the distributions.
        
        # Here is a second context
    
        cd_phone1 = [.2, .8]
        cd_phone2 = [.3, .7]
        cd_no_phone1 = [.4, .6]
        cd_no_phone2 = [.5, .5]
        
        ctx_phone_cd_args = {
            'story':"""
                An online survey company puts out a poll asking people two questions. 
                First, it asks if they buy physical CDs. Second, it asks whether they 
                own a smartphone. The company wants to determine if buying physical 
                CDs depends on owning a smartphone.
                """,
                's_sizes':[random.randint(40, 100), random.randint(10, 50)],
                'rows':['Smartphone', 'No smartphone'],
                'cols':['CD', 'No CD'],
                'row_dists':[random.choice([cd_phone1, cd_phone2]), 
                             random.choice([cd_no_phone1, cd_no_phone2])]
        }
        
        ctx_phone_cd = Chi2TestOfHomogeneityData(seed = seed, 
                                                  **ctx_phone_cd_args)
            
        # A default context where an initial set of observations is given
        # instead of the row distributions.
        Men = random.randint(35, 45)
        ctx_gender_math_args = {
            'story':"""
                A survey was given to 85 students in a Basic Algebra course, 
                with the following responses to the statement "I enjoy math."
                
                Test whether the distributions of responses to the survey
                are the same between men and women.
                """,
            'data':[[9,13,5,4,2],[12,18,11,6,5]],    
            'rows':['Men', 'Women'],
            'cols':['Strongly Agree', 'Agree', 'Nuetral', 'Disagree', 
                'Strongly Disagree'],
            's_sizes':[Men, 85 - Men]
        }

        ctx_gender_math = Chi2TestOfHomogeneityData(seed = seed,
                        **ctx_gender_math_args)
            
        return [ctx, ctx_phone_cd, ctx_gender_math] # , ctx_phone_cd]

    
    prob = Chi2TestOfHomogeneity(seed = seed)
    
    pb = ""
    for q_type in ['STAT', 'PVAL', 'HT']:
        for c in gen_ctx():
            result = prob.stem(context = c, q_type = q_type,
                               a_type = a_type)
            if result is not None:
               
                if a_type == 'preview':
                    pb += '<div class = \'posts\'>'
                    
                pb += result
                
                if a_type == 'preview':
                    pb += '</div><br>'
    print(pb)