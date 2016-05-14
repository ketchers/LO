import tools
import numpy as np
import sympy as sym
import random
import itertools as iter 
from functools import reduce
from sympy.functions.combinatorial.numbers import nC, nP, nT


class Rearrangements():
    """
    Evalauate rearrangements. 
    """
    
    def __init__(self):
    
        # A list to keep track of what has been done.
        self.done = []
    
    def stem(self, objects = None, a_type = None):
        """
        Here we will rearrange objects which are not all distinct.
        
        Named Parameters:
            objects -- These are the type of objects being rearranged. ('words', 'marbles', 'wordlike').
                       'wordlike' is really just like marbles, just a sequence of letters instead of marbles. 
                       The difference is just in how the question can be asked.
            a_type  -- The answer type, either "FR" (free response) or "MC" (multiple choice)
        """
        
        kwargs = {
            'objects': objects,
            'a_type': a_type
        }
        
        # Throw away problems whos answers are bigger than THREASH, but keep them in the done list
        INF = float('inf') # For now just take anything
        THREASH = 5000000 # I unset this for the 'words' type 
        
        # ERR_THREASH -- Throw awway error options such that (error - answer) > ERR_THREASH*answer
        ERR_THREASH = 4
        
        # These are the current possible values of the objects option
        OBJECT_TYPES = ["words", "marbles","wordlike"]
        
        if objects == None:
            objects = random.choice(OBJECT_TYPES)
            
        if a_type == None:
            a_type = random.choice(["FR", "MC"])
        
        # Some words with repeating letters
        WORDS = ["mathematics", "sleepless", "senseless", "carelessness"]
        WORDS += ["sleeplessness", "bubblebath", "senescence", "tweedledee"]
        WORDS += ["senselessness", "massless", "scissors", "pulchritude"]
        WORDS += ["losslessness", "inhibition", "knickknack","sweettooth" ]
        
        COLORS = ["red", "green", "blue", "orange", "yellow", "pink", "clear"]
        
        question_stem = ""
        
        def make_denom(l):
            """
            Takes a list [3,4,5] and outputs a string "3!4!5!" and the corresponding value.
            
            Parameters:
                l -- A list of integers
                
            Returns -- A pair (str, int) representing the denominator string representation
                       and value.
            """
            # The denominator in the permutation formula
            l = [i for i in l if i != 1]
            ls = map(str, l)
            denom_string = '!\\,'.join(ls) + "!"
            denom_val = reduce(lambda x,y: x*y, map(sym.factorial, l))
            return denom_string, denom_val
            
        def make_errors(a, n, l):
            """
            This could probably be improved. For now it generates some alternate answers for
            multiple choice, that on't look ridiculous.
            
            Parameters:
                a -- is the correct answer
                n -- is the number of actual objects
                l -- is the list of group sizes
            """
                
            l_ = l[:]
            errors = []
            while sum(l_) < n:
                l_.append(1)
            l_ = sorted(l_, key  = lambda x: -x)
            for k in range(len(l_)):
                for i in [-1, 1]:
                    for j in [-1, 1, 0]:
                        
                        if l_[k] + j > 1:
                            l_[k] = l_[k] + j
                            _, denom_val = make_denom(l_)
                            ans = sym.factorial(max(sum(l_), n + i)) / denom_val
                            if ans not in errors and ans != a and np.abs(a - ans) < ERR_THREASH * a:
                            #if ans not in errors and ans != a:
                                errors.append(ans)
            for i in [0.2, 0.3]:
                errors.append(int(a*(1 + i)))
                errors.append(int(a*(1-i)))
            random.shuffle(errors)
            return errors[0:4]
                
        
        def parse_word(word):
            """
            This convers a word into a dictionary of letters and counts and also 
            produces a sentence describing the situation.

            Examples: 

            parse_word("teepee) returns: ({'e': 4}, "There are 4 e's in teepee", [4])
            parse_word("sleeplessness") returns ({'e': 4, 'l': 2, 's': 5},
                                                 "There are 5 s's, 4 e's, and 2 l's in sleeplessness",
                                                 [5, 4, 2])

            
            Parameters:
            
                word -- a word
                
            Returns -- (dict, string, list) see above for example
            """
            word_dict = {}
            for k,g in iter.groupby(sorted(word), lambda x: x):
                count = len(list(g))
                if count > 1:
                    word_dict[k] = count
            l = ["%s %s's" % (word_dict[key],  key) for key in word_dict]
            
            word_dict_string = "There are " + tools.serialize(*l) + " in \"" + word + "\""
            word_letter_counts = list(word_dict.values())
            
            
            return word_dict, word_dict_string, word_letter_counts
        
        if objects == None:
            objects = random.choice(OBJECT_TYPES)
            
        
    
        if objects == "words": 
            
            THREASH = INF #This should not ba set for actual words
            
            word = random.choice(WORDS)
            
            num_objects = len(word)
            
            word_dict, word_string, group_counts = parse_word(word)
            
            item = (objects, group_counts)
            if item in self.done:
                return self.stem(**kwargs)
            
            self.done.append(item)
            
            denom_string, denom_val = make_denom(group_counts)
            
            answer = nP(word, len(word))
            
            question_stem = "How many distinct rearrangements of the letters in \"" + word  \
                + "\" are there?"
                
            explanation = "%s. So the number of distinct rearrangements of %s is:" % (word_string, word)
                
            
            explanation += "$$\\frac{%s!}{%s} = %s$$" %(len(word), denom_string, answer)
           
           
        elif objects == "marbles":
            
            
            # Get case correct
            def marble_string(i):
                if i == 1:
                    return "marble"
                else:
                    return "marbles"

            random.shuffle(COLORS)
            # You will have 3 - 5 groups with the distribution indicated
            num_groups = random.choice([3,3,4,4,4,5])
            # Each group will have between 1 to 5 members
            group_counts = [random.choice([2,2,3,3,3,4,4,5]) for i in range(num_groups)] 
            num_objects = sum(group_counts)
            
            marble_dict = dict([(COLORS[i], group_counts[i]) for i in range(len(group_counts))])
            
            print marble_dict, num_objects
            
            item = (objects, sorted(group_counts))
            if item in self.done:
                return self.stem(**kwargs)
            
            self.done.append(item)
            
            marble_string = tools.serialize(*["%s %s %s" % (group_counts[i], COLORS[i],                                     marble_string(group_counts[i])) for i in range(num_groups)])

            denom_string, denom_val = make_denom(group_counts)
            
            answer = nP(marble_dict, num_objects)
            
            # If number is too big try again
            if answer > THREASH:
                return self.stem(**kwargs)
            
            question_stem = "How many distint ways are there to arrange %s in a row?" % (marble_string)

            explanation = "The number of distinct rearrangements of the marbles is:"
            
            explanation += "$$\\frac{%s!}{%s} = %s$$" %(num_objects, denom_string, answer)

        elif objects == 'wordlike':
            
            
            # This code is repeated ... yuck
           
            # You will have 3 - 5 groups with the distribution indicated
            num_groups = random.choice([3,3,4,4,4,5])
            # Each group will have between 1 to 5 members
            group_counts = [random.choice([1] + [2]*2 + [3]*3 + [4]*2 + [5]) for i in range(num_groups)] 
            num_objects = sum(group_counts)
            
            item = (objects, sorted(group_counts))
            if item in self.done:
                return self.stem(**kwargs)
            
            self.done.append(item)
            
            def build_string(group_counts):
                alph = list('ABCDEFGHIJKLMNIPQRSTUVWXYZ')
                random.shuffle(alph)
                strg = []
                for i in group_counts:
                    strg += alph.pop()*i
                random.shuffle(strg)
                return ''.join(strg)

            strg = build_string(group_counts)
        
            denom_string, denom_val = make_denom(group_counts)
            
            answer = nP(strg, len(strg))
            
            # If number is too big try again
            if answer > THREASH:
                return self.stem(**kwargs)
            
            question_stem = "How many distint rearrangements of the string \'%s\' are there?"  % (strg)

            explanation = "The number of distinct rearrangements of the string \'%s\' is:"  % (strg)
            
            explanation += "$$\\frac{%s!}{%s} = %s$$" %(num_objects, denom_string, answer)
            
        else:
            print "Invald a_type: " + a_type 
           
        if a_type == "FR":
            question_stem += " Give your answer as an integer."
            answer_mathml = tools.fraction_mml(answer)
            return tools.fully_formatted_question(question_stem, explanation, answer_mathml)
        else:
            errors = make_errors(answer, num_objects, group_counts)
            distractors = [answer] + errors
            distractors = ["$_%s$_" % sym.latex(distractor) for distractor in distractors]
            return tools.fully_formatted_question(question_stem, explanation, answer_choices=distractors)
        # stem
            
       
        
if __name__ == "__main__":
    
    prob = Rearrangements()
    
    print '<h1> Testing objects = \'words\', a_type = \'MC\'</h1><br>'
    
    for i in range(4):
        print prob.stem(objects = 'words', a_type = 'MC')
    
    
    print '<br><h1> Testing objects = \'marbles\', a_type = \'MC\'</h1>'
    
    for i in range(4):
        print prob.stem(objects = 'marbles', a_type = 'MC')
        
    print '<br><h1> Testing objects = \'wordlike\', a_type = \'MC\'</h1>'
    
    for i in range(4):
        print prob.stem(objects = 'wordlike', a_type = 'MC')
    
