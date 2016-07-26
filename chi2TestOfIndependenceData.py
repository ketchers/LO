from __future__  import division
from __future__ import print_function
import random
import sys
import os
import numpy as np
import matplotlib as mpl
import scipy as sp
import scipy.stats as stats
import pylab as plt
import tools
import warnings
from table import Table

class DictObj(object):
    def __init__(self, dic):
        self.__dict__.update(dic)
    
def dict_to_obj(dic):
    return DictObj(dic)

# You need to run the previous cell first!
class Chi2TestOfIndependenceData(object):
    """
    This is the data for a chi-square independence test. If initialized as
      ctx = Chi2Independence(None)
    then a default is used. The default is silly just to indicate what can be 
    done. There will be three distributions with 4 outcomes. On each invocation
    a sample will be draw from each distribution to fill out the table. The 
    question is "Is outcome i independent of which distribution it is drawn 
    from?" Clearly there is a lot of flexibility in setting up such problems.
    
    
    In general the following parameters should be passed by name:
    
    
    Parameters:
    ----------
    seed    : None or int
        This is just to make the geeneration somewhat repeatable.
    story   : String
        A description of the context for the problem.
    row_name: String
        This is a general desctiption of the rows. For example "age".
    rows    : ['row header 1', 'row header 2', ...]
        These are the row headers for the tables. For example ['19 - 25', ...]
    row_name: String
        This is a general description of columns. For example "food preference"
    cols    : ['col header 1', 'col header 2', ...]
        These are the column headers for the tables. For example ['pizza', ...]
    s_sizes   : [int, int, ...] (length = num rows)
        Sample sizes for each row.
    row_dists : ['distribution', 'distribution', ...] (length = number of rows)
        row_dists[i] is the ditribution from which a sample of size s_size[i]
        will be drawn for the observed data. Either data or row_dists 
        should be initialized as None.
    data      : array of ints
        This wouls be an actual set of obsevered values. Either data or 
        row_dists should start as None.
    
    """
   
    
    def __init__(self, seed = None, **kwargs):
        
        if seed is not None:
            random.seed(seed) # Get predictable random behavior:)
            np.random.seed(seed)        
        
        context = dict_to_obj(kwargs)
        
        # Default context
        col_dict = {'colors':['red','blue','green','orange'],
            'foods':['pizza', 'burger','hotdog','sushi']}

        row_dict = {'weight':['140 - 160', '160 - 180', '180 - 200'],
                    'age':['25 - 31', '32 - 38', '38 - 44']}
        
        col_name = random.choice(col_dict.keys())
        cols = col_dict[col_name]
       
        
        row_name = random.choice(row_dict.keys())
        rows = row_dict[row_name]
        
        # Just a few dists for the default case
        d1 = np.ones(4) * 1/4 # Uniform
        d2 = np.array([.1,.3,.4,.2]) # Some "bell shaped" dist
        d3 = np.array([.4,.1,.1,.4]) # Some dist weighted toward the ends
        d4 = np.array([.5,.3,.1,.1 ]) # Right skewed
        # We will use the first 3 dists
        dists = [d1]*4  + [d2] + [d3] + [d4]
        random.shuffle(dists)
        
        a_level = random.choice([0.01, 0.05, 0.1])
        
        story = """
                A group of adults was asked about their preference in %s. The
                participants were put into categories by %s. Test at the level
                $_\\alpha = %.2f$_ whether %s and %s are independent.
                """%(col_name, row_name, a_level, col_name, row_name)
        # End default context
                
        self.cols = getattr(context, 'cols', cols)
        self.rows = getattr(context, 'rows', rows)
        
        # Choose sizes in each category
        self.s_sizes = getattr(context, 's_sizes', 
                        [random.randint(20,60) for i in range(len(self.rows))])
        

        data = getattr(context, 'data', None)
        row_dists = getattr(context, 'row_dists', None)
        if data is not None:
            data, row_dists = self.gen_data(data = data, row_dists = None)
        elif row_dists is not None:
            data, row_dists = self.gen_data(data = None, row_dists = row_dists)
        else:
            data, row_dists = self.gen_data(data = None, 
                                       row_dists = dists[:len(self.rows)])
        
        self.row_dists = row_dists
        self.observed = data
        self.obs_marg = self.add_marginals(self.observed)
                
        
        
        self.row_name = getattr(context, 'row_name', row_name)
        self.col_name = getattr(context, 'col_name', col_name)
        
        self.story = getattr(context, 'story', story)        
        
        null = "%s and %s are independent." % (self.col_name, self.row_name)
        
        alternative = "%s and %s are dependent." % (self.col_name, 
                                                    self.row_name)
    
        self.null = getattr(context, 'null', null)
        self.alternative = getattr(context, 'alternative', alternative)
       
        self.a_level = getattr(context, 'a_level', a_level)
        
        self.probs = self.obs_marg / self.obs_marg[-1,-1]
        
        self.expected = self.obs_marg[:,[-1]]\
            .dot(self.obs_marg[[-1],:]) / self.obs_marg[-1,-1]
                            
        self.chi2_stat = self.compute_chi2(self.observed)
        
        self.df = (data.shape[0] - 1) * (data.shape[1] - 1)        
                
        self.note = getattr(context,'note',"") 
        
        self.observed_table = Table(self.observed, row_headers = self.rows,
                                    col_headers = self.cols)
       
        self.obs_marg_table = Table(self.obs_marg, 
                                    row_headers = self.rows + ['Total'],
                                    col_headers = self.cols + ['Total'])

        self.expected_table = Table(self.expected, 
                                    row_headers = self.rows + ['Total'],
                                    col_headers = self.cols + ['Total'])
        

        self.hash = 17
        

    def gen_data(self, row_dists = None, data = None, threash = 1.0):
        """
        There are two opions
        
        (1) User supplied data. In this case row_dists are generted from 
        supplied data and these are used for sampling.
        
        (2) User supplied row_dists. In this case these are used for sampling.
        
        In either case validity is checked, based on THREASH and if the data 
        is invalid, a new set is generated.
        
        
        Parameters:
        ----------
        row_dists : ['row1 distribution', 'row2 distribution', ...]
            A distribution is a sequence of non-negative numbers 
            which sum to 1.
        data      : array of shape num_rows x num_cols 
                    (list of lists or numpy array)
            This would be an actual set of obseved values. From this the 
            row_dists will be determined and in each problem a new set of data
            will be produced based on these dists.
        """
        if data is not None:
            # It is assumed in this case that row_dists  is None
            data = np.array(data)
            row_dists = data / data.sum(axis = 1).reshape(data.shape[0], 1)
        
        # it is assumed here that now row_dists is not None
    
        data = [np.random.choice(self.cols, (1, self.s_sizes[i]), 
                                 p = row_dists[i]).flatten() 
                                     for i in range(len(self.rows))]
                                         
                
        data = np.array([[np.sum(dat == i) for i in self.cols] for dat in data])
        
        if not self.is_valid(threash, data):
            warnings.warn("The cell counts are too small! Regenerating dataset.")
            return self.gen_data(row_dists = row_dists, data = None)
        
        return data, row_dists
        
    def is_valid(self, threashold, data):
        return np.sum(data > 4)/data.size >= threashold and np.all(data > 0)

    def get_counts(self, data):
        return np.array([[np.sum(trial == i) \
             for i in range(len(self.cols)*len(self.rows))] \
                      for trial in data])
    
    def add_marginals(self, data):
        data_marg = np.vstack([data.T,np.sum(data, axis = 1)]).T 
        data_marg = np.array(data_marg)
        data_marg = np.vstack([data_marg, data_marg.sum(axis=0)])
        return data_marg

    def compute_chi2(self, a):
        a_marg = self.add_marginals(a)
        N = a_marg[-1,-1]
        O = a
        E = a_marg[:,[-1]].dot(a_marg[[-1],:]) / N
        E = E[:-1,:-1]
        chi2_stat = np.sum((E - O)**2/E)
        return chi2_stat
        
    def show(self, path = "", fname = None, force = False):
        
        if fname != 'show':
            fname = path + "/" + str(hash(self)) + "_plot.png"
            
        
        if fname is not 'show' and os.path.isfile(fname) and not force:
            print("The file \'" + fname + "\' exists, \
            not regenerating. Delete file to force regeneration.", file=sys.stderr)
            return fname        
        
        
        df = self.df
        a_level = self.a_level
        chi2_stat = self.chi2_stat
        cols = self.cols
        rows = self.rows
        row_dists = self.row_dists
        s_sizes = self.s_sizes
        observed = self.observed
        obs_marg = self.obs_marg
        N = obs_marg[-1,-1]
        exp_dists = self.expected[:-1,:-1] / N
        
        rv = stats.chi2(df)
        
        x_data = np.linspace(max([0, rv.mean() - 3*rv.std()]), 
                             max([8, rv.mean( ) + 4*rv.std()]), 200)
                             
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
             
        
        ax.text((min(x_data) + max(x_data)) / 2 , 
                (min(y_data) + min([1,max(y_data)])) / 2,
               "p-value = %.4g%s" % ((1-rv.cdf(chi2_stat)) * 100, '%'), 
                fontsize = 14)
         
        
        ax.plot([chi2_stat, chi2_stat], [0, max(.005,rv.pdf(chi2_stat))], 'r-', 
                lw = 2)
        ax.fill_between(x_data[x_above_crit ], 0, y_data[x_above_crit],
                        color = (.7,.2,.7,.5))
        ax.fill_between(x_data[x_above_statistic], 0, y_data[x_above_statistic],
                        color = (.7,.2,.7,.3))
    
        
        shape = observed.shape
    
        samples = np.random.choice(range(len(cols)*len(rows)), (5000, N), 
                                   p = exp_dists.flatten())
                                   
        samples = self.get_counts(samples)
        
        samples = samples.reshape(5000, shape[0], shape[1])
    
        sample_chi2 = np.array([self.compute_chi2(a) for a in samples])        
        
       
        q = sample_chi2
       
        q = q[q < rv.mean( )+ 4*rv.std()]
        ax.hist(q, bins=15, normed=True, color = (.8,.8,1,.2), 
                histtype='stepfilled', lw=1, ls=":")
              
        if fname == 'show':
            plt.show()            
        else:
            tools.make_folder_if_necessary(".", path)        
            plt.savefig(fname) 
        plt.close()
        return fname
        
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
            
if __name__ == "__main__":
    
    
    # Basically a default context with fixed row sizes and distribuions
    ctx = Chi2TestOfIndependenceData(s_sizes = [30,30,30],
                                     data = [[6,6,6,12],[12,6,6,6],[10, 10, 8, 2]],
                                     seed = 42);
    
    
    
    # Here is a second context
    story = """
    An online survey company puts out a poll asking people two questions. 
    First, it asks if they buy physical CDs. Second, it asks whether they 
    own a smartphone. The company wants to determine if buying physical 
    CDs depends on owning a smartphone.
    """

    cd_phone1 = [.2, .8]
    cd_phone2 = [.3, .7]
    cd_no_phone1 = [.4, .6]
    cd_no_phone2 = [.5, .5]
    
    s_sizes = [random.randint(40, 100), random.randint(10, 50)]
    
    rows = ['Smartphone', 'No smartphone']
    cols = ['CD', 'No CD']
    
    row_dists = [random.choice([cd_phone1, cd_phone2]), 
                 random.choice([cd_no_phone1, cd_no_phone2])]
    ctx_phone_cd = Chi2TestOfIndependenceData(story = story, 
                    rows = rows, 
                    cols = cols, 
                    s_sizes = s_sizes, 
                    row_dists = row_dists,
                    seed = 42)
     

    ctx.show(fname = 'show')
    print(ctx.story)   
    
    
    print(Table.get_style())
    print(ctx.obs_marg_table.html())
    print(ctx.expected_table.html())
    print(ctx.obs_marg_table.latex())
    print(ctx.expected_table.latex())

            
    #ctx_phone_cd.show(fname = 'show')
    #print(ctx_phone_cd.story)
    