from __future__  import division
from __future__ import print_function
import random
import numpy as np
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
   
    
    def __init__(self, **kwargs):
        
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
        self.sample = np.random.choice(self.outcomes, self.s_size, 
                                       p = self.o_dist)
        # Generate counts
        self.o_counts = np.array([sum(self.sample == i) for i in self.outcomes])
        # Sample distribution
        
        self.is_valid = sum(self.o_counts - 4 > 0)/len(self.o_counts) >= .8 \
            and all(self.o_counts > 0)
        if not self.is_valid:
            warnings.warn("The cell counts are too small!")
        
        
        self.s_dist = self.o_counts/sum(self.o_counts)
        # Set alpha level (a_level)
        self.a_level = getattr(context, 'a_level', random.choice([.1,.05,.01]))
        # The actual description of the problem
        self.chi2_stat = np.sum((self.o_counts - self.t_counts)**2/self.t_counts)
        self.df = len(self.o_counts)  - 1        
        self.null = getattr(context, 'null', "The die is fair with each outcome \
                being equally likly.")
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
        
    def __hash__(self):
        if self.hash == 17:
            hlist = map(lambda x: tuple(x) if type(x) in [list, np.ndarray] else x, 
                        [self.__dict__[i] for i in sorted(self.__dict__)])
            for i in hlist:
                self.hash = hash(hash(hlist[i]) + 31 * self.hash)
            return self.hash
        return self.hash
        
    def __eq__ (self, other):
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__
        
    def __neq__ (self, other):
        return not self.__eq__(other)
            
       