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
from chi2GoodnessOfFitData import Chi2GoodnessOfFitData

class Chi2GoodnessOfFit(object):
    
    def __init__(self, seed = None, path = "chi2gof"):
        self.cache = set() #Just for names this time.
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
            np.random.seed(seed)
        self.hist = ""
        self.solution_plot = ""
        self.path = path
        
        
    def stem(self, context = None, table = None, q_type = None,
             a_type = 'preview', force = False, fmt = 'html'):
        """
        This will generate a problem for $\chi^2$ goodness of fit.
        
        Parameters:
        ----------
        context : context object
            This describes the problem. A default context is used if this is 
            none.
        table   : string ['hist', 'table'] 
            Display t_dist as a table (html/latex) or as a histogram.
        q_type  : string [None, 'STAT', 'HT', 'CI'] 
            If None randomly choose. If 'STAT' just compute the chi2 statistic 
            and the degrees of freedom. If 'HT, compute the p-value for the  
            data and determine whether or not reject the null hypothesis. If 
            'CI' compute the confidence interval.
        a_type  : string
            This is eithe "MC" or "preview" for now
        fmt   : String ['html', 'latex']
            Use nice CSS/HTML (responsive) or plain LaTeX (static)
        
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
            'fmt': fmt,
            'a_type': a_type
        }
        
        if q_type is None:
            q_type = random.choice(['STAT', 'HT', 'PVAL'])
          

        if table == None:
            table = random.choice(['table', 'hist'])
        
        if context == None:
            context = Chi2GoodnessOfFitData()
            
        if not context.is_valid:
            warnings.warn("Context had invalid cell counts.")
            return       
        
        # Generate unique name
        q_name = hash(context)
        self.cache.add(q_name)
        self.hist = str(q_name) + "_hist.png"
        self.solution_plot = str(q_name) + "_plot.png"
        
        style = None
            
        
        if a_type == 'preview':
                question_stem = "<h2>Question</h2><br>"
        else:
            question_stem = ""

        if fmt == 'html':
            question_stem += Table.get_style()
        
        question_stem += "<div class='par'>" + context.story + "</div>\n" 
        if table == 'table':
            if fmt == 'html':
                tbl = context.observed.html()
                question_stem += tbl
            else:
                tbl = context.observed.latex()
                question_stem +=  tbl
            
        elif table == 'hist':
            
            fname = context.hist(path = self.path, force = force)
                                  
            img = html_image(fname, width = '300px', 
                             preview = (a_type == 'preview'))
            
            
            if style is None:
                style = Table.get_style()
                question_stem += style
            
            question_stem +="""      
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
            """ % (img, "Observed Frequencies")                
        
 
        N = len(context.outcomes)     
        df =  N - 1        
        chi2eq = "$$\\chi^2_{%s}=\\sum_{i=1}^{%s}\\frac{(O_i-E_i)^2}{E_i}\
                = %.3g$$" % (df, N, context.chi2_stat)
                
        if q_type == 'STAT':
            question_stem += "Compute the $_\\chi^2$_-statistic and degrees \
                of freedom for the given observed values."

        elif q_type == 'PVAL':
           
            question_stem += """The $_\\chi^2$_ statistic is 
                {chi2eq}
                                 
                
            Use this information to find the degrees of freedom (df) and the 
            $_p\\text{{-value}} = P(\\chi^2_{{{df}}} > {chi2:.3g})$_.
            """.format(chi2eq=chi2eq, N=N, df = 'df', chi2=context.chi2_stat)
                       
        elif q_type == 'HT':
            
            question_stem += """The degrees of freedom are $_df = {N} - 1 = 
            {df}$_ and the $_\\chi^2$_ statistic is {chi2eq} 
            
            Use this information to conduct a hypothesis test with 
            $_\\alpha = {a_level}$_. Choose the answer that best captures
            the null hypothesis and conclusion.
            """.format(N = N, df = df, chi2eq = chi2eq,
                       a_level = context.a_level)
       
        if fmt == 'html':
            explanation = Table.get_style()
        else:
            explanation = ""
        
        if a_type == 'preview':
            explanation += "<br><h2>Explanation</h2><br>"
            
        tb1 = Table(context.t_dist, col_headers = context.outcomes,
                    row_headers = [context.outcome_type, 'Probabilities'])
                    
        if fmt == 'html':                    
                        
            tbl1 = tb1.html()
            tbl2 = context.oe.html()       
        else:
            tbl1 = tb1.latex()
            tbl2 = context.oe.latex()   
        
        explanation += "<div class='par'>To find the expected counts multiply the total\
            number of observations by the expected probability for an outcome. \
            The probabilities for the expected outcomes are summarized in the \
            following table:"
        explanation += tbl1
        explanation += "and there are %s observations." % context.s_size
        explanation += " So the expected and observed counts are:<br>"
        explanation += tbl2 
        explanation += "</div>"
        
        explanation += "<div class='par'>The degrees of freedom are $_df = {N} - 1 = \
        {df}$_ and the $_\\chi^2$_-statistic is:{chi2eq}</div>"\
                .format(N=N,df=df,chi2eq=chi2eq)
        
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
                    <div class='par'>The p-value is less than the 
                    $_\\alpha$_-level so the null hypothesis is rejected. 
                    That is, we accept the alternative hypothesis:
                    <strong>H<sub>a</sub>: {alt} </strong>
                
                    More precisely, assuming the null hypothesis, there
                    is only a {p_val:.2%} probability due to 
                    random chance in sampling that
                    the difference in the expected and observed data is 
                    least this large.</div>
                    """.format(alt=context.alternative, p_val=p_val)
            else:
                explanation += """
                    <div class='par'>The p-value is greater than the $_\\alpha$_-level so
                    the null hypothesis
                    is not rejected. Precisely, assuming the null hypothesis
                    <strong>H<sub>0</sub>: {null}</strong>,
                    there is a {p_val:.2%} probability due to 
                    random chance in sampling that
                    the difference in the expected and observed data is at 
                    least this large.</div>
                    """.format(null=context.null, p_val=p_val)
                    
            explanation += """
                <div class='par'>Note: {note}</div>                
                """.format(note=context.note)
        
        
        errors = self.gen_errors(q_type, context)
        
        if a_type == 'preview':
            errs = [[er] for er in errors]
            choices = "<br><h2>Choices</h2><br>"
            tb = Table(errs, row_headers = ['Answer'] + ['Distractor']*4)
         
            # Choices need the richer html structure in preview.
            choices += Table.get_style() + tb.html()   
                        
            if fmt == 'html':
                return question_stem + choices +  explanation
            else:
                return question_stem.replace("<div ","<p ")\
                        .replace("</div>","</p>") + choices + \
                        explanation.replace("<div ","<p ")\
                        .replace("</div>","</p>")
                        
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
        N = len(context.outcomes)
        df = N - 1
        # A few potential errors
        # df = N instead of N - 1
        # use |O_i  - E_i|/E_i instead of (O_i - E_i)^2
        # use |O_i  - E_i|/E_i instead of (O_i - E_i)^2 and df = N
        # use (O_i - E_i)/O_i
        # use (O_i - E_i)/O_i and df = N
        # take sqrt of chi^2-stat
           
        errors = [(N, context.chi2_stat),
                  (df, np.sum(np.abs(context.t_counts - 
                      context.o_counts) / context.t_counts)),
                  (N, np.sum(np.abs(context.t_counts - 
                      context.o_counts) / context.t_counts)),
                  (df, np.sum((context.t_counts - 
                       context.o_counts)**2 / context.o_counts)),
                  (N, np.sum((context.t_counts - 
                      context.o_counts)**2 / context.o_counts)),
                  (df, context.chi2_stat ** .5)]
                  
        if q_type == 'STAT':
        
            def error_string0(df, chi2):
                ans = '(degrees of freedom) df = %s and the $_\\chi^2$_-test statistic = %.3g'\
                       % (df, chi2)
                return ans
                
            ans = error_string0(df, context.chi2_stat)
            errors = map(lambda x: error_string0(*x), errors)
            
            
            
            
        if q_type in ['PVAL']:
            
            def error_string1(df, chi2):
                rv = stats.chi2(df)
                p_val = 1 - rv.cdf(chi2)
                
                ans = '(degrees of freedom) df = %s and the p-value = %.3g'\
                       % (df, 1 - p_val)
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
                        $_\\alpha$_-level of {a_level:.0%}. Therefore we reject the
                        null hypothesis and find evidence in favor of the 
                        alternative hypothesis: 
                        <strong>H<sub>1</sub>: {alt}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   alt = context.alternative)
                elif p_val >= a_level and correct:
                     ans = """
                        The p-value is {p_val:.2%} and this is greater than the 
                        $_\\alpha$_-level of {a_level:.0%}. Therefore we fail to 
                        reject the null hypothesis thus supporting the 
                        hypothesis: 
                        <strong>H<sub>0</sub>: {null}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   null = context.null)
                                   
                elif p_val < a_level and not correct:
                    ans = """
                        The p-value is {p_val:.2%} and this is less than the 
                        $_\\alpha$_-level of {a_level:.0%}. Therefore we fail to 
                        reject the null hypothesis thus supporting the 
                        hypothesis: 
                        <strong>H<sub>0</sub>: {null}</strong>
                        """.format(p_val = p_val, a_level = a_level, 
                                   null = context.null)
                else:
                     ans = """
                        The p-value is {p_val:.2%} and this is greater than the 
                        $_\\alpha$_-level of {a_level:.0%}. Therefore we reject the
                        null hypothesis and find evidence in favor of the 
                        alternative hypothesis: 
                        <strong>H<sub>1</sub>: {alt}</strong>
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
    
    a_type = 'MC'
    force = False
    fmt = 'latex'
    seed = 42
      
    def gen_ctx():
        ctx = Chi2GoodnessOfFitData(seed = seed) 
    
        #Here we sample from a non-uniform distribution for the die!
        ctx1_args = {
            'o_dist':[1/5, 1/5, 1/5, 1/5, 1/10, 1/10],
            'alternative':"The die is not fair.",
            'note':"""
                For this problem the truth is tha the die is not fair. \
                If you accepted H<sub>0</sub>, then this is a <strong>miss</strong>
                (Type II error).
                """
        }
        
        ctx1 = Chi2GoodnessOfFitData(seed = seed, **ctx1_args)
        
                
        ######################################
        # Pass the pigs game: Due to embedding the table in the story
        # this one is a little harder to set up. The problem is that this 
        # pollutes namespace for future contexts. 
        
         
        # Append _ to avoid namespace pollution
        _outcomes =  ['Pink', 'Dot', 'Razorback', 'Trotter', 
                     'Snouter', 'Leaning Jowler']
        _t_dist = [.35, .30, .20, .10, .04, .01]
        tb = Table(_t_dist, col_headers = _outcomes, 
                   row_headers = ['Position', 'Expected Frequency'])
        
        if fmt == 'html':
            styles = Table.get_style()
            tbl = tb.html()
        else:
            styles = ""
            tbl = tb.latex()
        
        _s_size = random.randint(20, 30) * 10
        
        ctx2_args = {
            'outcome_type':'Position',
            'outcomes':_outcomes,
            't_dist':_t_dist,
            's_size':_s_size,
            'story':"""Pass The Pigs&reg; is a game from Milton-Bradley&#8482; 
                which is essentially a dice game, except that instead of dice 
                players toss small plastic pigs that can land in any of 6 
                positions. For example, you roll a 'trotter' if the pig falls 
                standing on all 4 legs. It is claimed that the distribution
                for the 6 positions are:
                
                {styles}
                {tbl}            
                
                To test this you toss a pig {s_size} times and get the observed 
                frequencies below:
                """.format(styles = styles, tbl = tbl, s_size = _s_size),
            'null':"The observed values of the positions of the pigs agrees \
            with the expected distribution.",
            'alternative':"The observed values of the positions of the pigs \
            differs from what should be the case if the expected distribution\
            was correct.",
            'note':"""
                In this case the observed data was sampled from the given
                distribution. So if the null hypothesis is rejected, this
                is <string>a Type-I error</strong> or <strong>a false 
                positive</strong>.
                """
        }
        
        ctx2 = Chi2GoodnessOfFitData(seed = seed, **ctx2_args)
        
       
            
        ###########################################
        ## 11.2 from text
        s_size = random.randint(5, 10) * 10
        ctx3_args = {
            'outcomes':['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday'],
            't_dist':np.ones(7) * 1/7,
            's_size':s_size,
            'a_level': random.choice([0.1,0.01,0.05]),
            'outcome_type':'Day of Week',
            'story':"""
                  Teachers want to know which night each week their students are 
                  doing most of their homework. Most teachers think that students 
                  do homework equally throughout the week. Suppose a random sample 
                  of %s students were asked on which night of the week they 
                  did the most homework. The results were distributed as in  
                  """ % s_size,
            'null':"Students are equally likely to do the majority of their \
            homework on any of the seven nights of the week.",
            'alternative':"Students are more likely to do the majority of their \
            homework on certain nights rather than others."
        }
            
        ctx3 = Chi2GoodnessOfFitData(seed = seed, **ctx3_args)
            
        return [ctx, ctx1, ctx2, ctx3]
            
    
    
    prob = Chi2GoodnessOfFit(seed = seed)
    pb = ""
    for q_type in ['STAT','PVAL','HT']:
        for table in ['hist', 'table']:
            for c in gen_ctx():
                result = prob.stem(context = c, table=table, a_type = a_type, 
                                   q_type = q_type, force = force, fmt = fmt)

                if result is not None:
                    if a_type == 'preview':
                        pb += '<div class = \'posts\'>'

                    pb += result

                    if a_type == 'preview':
                        pb += '</div><br>'
    print(pb)