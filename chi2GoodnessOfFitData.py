from __future__  import division
from __future__ import print_function
import random
import os 
import sys
import numpy as np
import matplotlib as mpl
import scipy as sp
import scipy.stats as stats
import pylab as plt
import tools
from table import Table
import warnings

class DictObj(object):
    def __init__(self, dic):
        self.__dict__.update(dic)
    
def dict_to_obj(dic):
    return DictObj(dic)

# You need to run the previous cell first!
class Chi2GoodnessOfFitData(object):
    """
    This is the data for a chi-square goodness of fit test. If initialized as
      ctx = Chi2GoodnessOfFit(None)
    then some defaults are used (see below). These can then be modified
      ctx.outcomes = [u'\u2680', u'\u2681', u'\u2682', u'\u2683', u'\u2684', u'\u2685']
    
    If initialized as:

    # Pass the pigs game
    outcomes =  ['Pink', 'Dot', 'Razorback', 'Trotter', 'Snouter', 'Leaning Jowler']
    t_dist = [.35, .30, .20, .10, .04, .01]
    
    tbl, styles = make_table(outcomes, ['Position', 'Expected Frequency'],True, 
                            t_dist)
    
    story = \"""Pass The Pigs&reg; is a game from Milton-Bradley&#8482; which is 
            essentially a dice game except that instead of dice players toss
            small plastic pigs that can land in any of 6 positions. For example, 
            you roll a trotter if the pig falls standind on all 4 legs. 
            The expected for the 6 positions are:
            
            {styles}
            {tbl}            
            \""".format(styles = styles, tbl = tbl)
            
    ctx2 = Chi2GoodnessOfFitData(
        outcomes = outcomes,
        t_dist = t_dist,
        s_size = random.randint(5, 20) * 6,
        a_level = random.choice([0.1,0.01,0.05]),
        story = story)
    
    """
   
    
    def __init__(self, seed = None, **kwargs):
        
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
            np.random.seed(seed)
        
        context = dict_to_obj(kwargs)
                        
        # We will have a default contex of testing for a fair die. The theoretical
        # distribution (t_dist) is [1/6,1/6,1/6,..]
        self.outcome_type = getattr(context,'outcome_type','Die')
        self.outcomes = getattr(context, 'outcomes', range(1,7))
        self.t_dist = np.array(getattr(context, 't_dist', 1/6.0*np.ones(6)))
        # We have a default sample size (s_size) divisible by 6 so that all 
        # counts are integers
        self.s_size = getattr(context, 's_size', random.randint(5, 20) * 6)
        # The expected counts 
        self.t_counts = getattr(context, 't_counts', self.t_dist * self.s_size)
        # An additional observed distribution may be given, otherwise this 
        # defaults to the theoretical distribution.
       
        
        self.o_dist = getattr(context, 'o_dist', self.t_dist)
        # Generate the actual sample.
        
        self.o_counts, self.sample  = self.gen_data(threshold = 1.0)        
        
        
        # Sample distribution
        self.expected = Table(self.t_counts, col_headers = self.outcomes,
                              row_headers = [self.outcome_type, 'Expected'])
        
        self.observed = Table(self.o_counts, col_headers = self.outcomes,
                              row_headers = [self.outcome_type, 'Observed'])
                              
        self.oe = Table(self.t_counts, self.o_counts, col_headers = self.outcomes,
                        row_headers = [self.outcome_type, 'Expected', 'Observed'])
                
        
        
        
        
        
        if not self.is_valid:
            warnings.warn("The cell counts are too small!")
        
        
        self.s_dist = self.o_counts/sum(self.o_counts)
        # Set alpha level (a_level)
        self.a_level = getattr(context, 'a_level', random.choice([.1,.05,.01]))
        # The actual description of the problem
        self.chi2_stat = np.sum((self.o_counts - self.t_counts)**2/self.t_counts)
        self.df = len(self.o_counts)  - 1        
        self.null = getattr(context, 'null', "The die is fair with each outcome \
                being equally likely.")
        self.alternative =  getattr(context, 'alternative', "The die is not fair some\
            outcomes are more likely than others.")
        self.note = getattr(context,'note',"""
            The sample here was taken from the given expected distribution.
            If you rejected the null, then this is a <strong>false 
            positive</strong> (Type I error). 
            """)
        

        default_story = """
            To test if a die is fair you roll the die
            {s_size} times with the following outcomes:
            """.format(s_size = self.s_size)


        self.story = getattr(context, 'story', default_story)
        self.hash = 17
        
    def gen_data(self, threshold = 1.0, count = 0):
        """
        This generates observed data from o_dist. A check of vlidity is made.
        
        Parameters:
        ----------
        count : int
            This is for internal use if it goes beyond 20 an error is thrown
            telling the user that either th probabilities are too small or 
            the sample size is too small to get valid cell counts.
        threashold : float between 0 and 1
            This says determine the percentage of cells that must have 
            count > 4. 
        """
        if count > 20:
            raise ValueError("Problem generating data with valid cell count.") 
            
        sample = np.random.choice(self.outcomes, self.s_size, 
                                       p = self.o_dist)
        # Generate counts
        data = np.array([sum(sample == i) for i in self.outcomes])
        if self.is_valid(threshold, data):
            return data, sample
            
        return self.gen_data(threshold = threshold, count = count + 1)
        
    def is_valid(self, threshold, data):
        return sum(data - 4 > 0)/len(data) >= threshold \
            and all(data > 0)   
        
        
    def __hash__(self):
        if self.hash == 17:
            ls = list(self.__dict__)
            ls.sort()
            ls = [(it, self.__dict__[it]) for it in ls]
            self.hash = np.abs(hash(repr(ls)))
        return self.hash
        
    def __eq__ (self, other):
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__
        
    def __neq__ (self, other):
        return not self.__eq__(other)
        
    def show(self, path = "", fname = None, force = False):
        
        if fname != 'show':
            fname = path + "/" + str(hash(self)) + "_plot.png"
        
        if fname is not 'show' and os.path.isfile(fname) and not force:
            print("The file \'" + fname + "\' exists, \
            not regenerating. Delete file to force regeneration.", file=sys.stderr)
            return fname
            
        chi2_stat = self.chi2_stat
        outcomes = self.outcomes
        t_dist = self.t_dist
        s_size = self.s_size
        a_level = self.a_level
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
                "p-value = %.4g%s" % ((1-rv.cdf(chi2_stat)) * 100, '%'), fontsize = 14)
        
        ax.plot([chi2_stat,chi2_stat], [0,max(.005,rv.pdf(chi2_stat))], 'r-', lw = 2)
        ax.fill_between(x_data[x_above_crit ], 0, y_data[x_above_crit],
                        color = (.7,.2,.7,.5))
        ax.fill_between(x_data[x_above_statistic], 0, y_data[x_above_statistic],
                        color = (.7,.2,.7,.3))
    
    
        s = np.random.choice(outcomes, (5000, s_size), p = t_dist)
        q = tuple(np.transpose([np.sum(s == i, axis = 1)]) for i in outcomes)
        q = np.sum((np.hstack(q) - 
                    s_size*np.ones((5000, len(t_dist)))*np.array(t_dist)) **2 / (s_size * np.array(t_dist))
                   , axis=1)
        q = q[q < rv.mean( )+ 4*rv.std()]
        ax.hist(q, bins=15, normed=True, color = (.8,.8,1,.2), histtype='stepfilled', lw=1, ls=":")
        
        if fname == 'show':
            plt.show()            
        else:
            tools.make_folder_if_necessary(".", path)        
            plt.savefig(fname) 
        plt.close()
        return fname
    
        
    def hist(self, rotation = None, path = "", fname = None, force = False):
        
        if fname != 'show':
            fname = path + "/" + str(hash(self)) + "_hist.png"
        
        if fname is not 'show' and os.path.isfile(fname) and not force:
            print("The file \'" + fname + "\' exists, \
            not regenerating. Delete file to force regeneration.", file=sys.stderr)
            return fname
        
        outcomes = self.outcomes
        outcome_dict = dict([(b, a) for a,b in enumerate(outcomes)])
        observed = [outcome_dict[a] for a in self.sample]
        
        s_size = self.s_size
        dist = self.o_dist
        
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
                    rotation = -45
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

        tools.make_folder_if_necessary(".", path)        
        if fname == 'show':
            plt.show()            
        else:
            tools.make_folder_if_necessary(".", path)        
            plt.savefig(fname)
        plt.close()
        return fname
        
  
        
        
if __name__ == "__main__":
    
    
    #Here we sample from a non-uniform distribution for the die!
    ctx1_args = {
        's_size':40,
        'o_dist':[1/5, 1/5, 1/5, 1/5, 1/10, 1/10],
        'alternative': "The die is not fair.",
        'note': """
                For this problem the truth is tha the die is not fair. \
                If you accepted H<sub>0</sub>, then this is a <strong>miss</strong>
                (Type II error).
                """
    }
    ctx1 = Chi2GoodnessOfFitData(seed = seed, **ctx1_args)
    ctx1.show(fname='show')
    ctx1.hist(fname='show')   
    print(ctx1.expected.latex())
    print(ctx1.expected.html())
    
    ctx = Chi2GoodnessOfFitData()
    ctx.show(fname='show')
    ctx.hist(fname='show')
   
    print(ctx.expected.latex())
    print(ctx.expected.html())

    print(ctx.observed.latex())
    print(ctx.observed.html())    
    
    