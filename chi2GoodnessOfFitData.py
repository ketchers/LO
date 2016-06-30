from __future__  import division
from __future__ import print_function
import random
import numpy as np

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
    
    If initialized as 
      ctxNew = Chi2GoodnessOfFit(ctx)
    then all the attributes of ctx are used. (A copy mechanism.)
    
    """
   
    
    def __init__(self, **kwargs):
        
        context = dict_to_obj(kwargs)
                        
        # We will have a default contex of testing for a fair die. The theoretical
        # distribution (t_dist) is [1/6,1/6,1/6,..]
        self.outcome_type = 'Die'
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
        self.o_counts = [sum(self.sample == i) for i in self.outcomes]
        # Sample distribution
        self.s_dist = self.o_counts/sum(self.o_counts)
        # Set alpha level (a_level)
        self.a_level = getattr(context, 'a_level', random.choice([.1,.05,.01]))
        # The actual description of the problem
        self.chi2_stat = np.sum((self.o_counts - self.t_counts)**2/self.t_counts)
        

        default_story = """
            To test if a die is fair you roll the die
            {s_size} times with the following outcomes:
            """.format(s_size = self.s_size)


        self.story = getattr(context, 'story', default_story)
        self.hash = 17
        
    def __hash__(self):
        if self.hash == 17:
            for i in sorted(self.__dict__):
                self.hash = hash(self.__dict__[i] + 31 * self.hash)
            return self.hash
        return self.hash
        
    def __eq__ (self, other):
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__
        
    def __neq__ (self, other):
        return not self.__eq__(other)
            
       