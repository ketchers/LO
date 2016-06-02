"""General tools for problem generation"""

import random
from sympy import *
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import distutils.spawn
x = symbols('x')


#if distutils.spawn.find_executable('itex2mml') != '/usr/local/bin/itex2mml':
#    print 'ERROR: brew install itex2mml'
#    sys.exit(-1)

NAMES_TUPLES = [("Jamie", "her", "she", "her"),
                ("John", "his", "he", "him"),
                ("Greg", "his", "he", "him"),
                ("Marc", "his", "he", "him"),
                ("Annie", "her", "she", "her"),
                ("Jessica", "her", "she", "her"),
                ("Alice", "her", "she", "her"),
                ("Jolyn", "her", "she", "her"),
                ("Porter", "his", "he", "him"),
                ("Ariana", "her", "she", "her"),
                ("Rosetta", "her", "she", "her"),
                ("Lexie", "her", "she", "her"),
                ("Horace", "him", "he", "him"),
                ("Hugo", "him", "he", "him"),
                ("Gail", "her", "she", "her"),
                ("Renee", "her", "she", "her"),
                ("Josslyn", "her", "she", "her"),
                ("Floretta", "her", "she", "her"),
                ("Evelyn", "her", "she", "her"),
                ("William", "his", "he", "him"),
                ("Daniel", "his", "he", "him"),
                ("Emily", "her", "she", "her"),
                ("Emma", "her", "she", "her"),
                ("Olivia", "her", "she", "her"),
                ("Isabella", "her", "she", "her")]

NAME_DICTS = []
for name_tuple in NAMES_TUPLES:
    new_name = {'name': name_tuple[0], 'possessive': name_tuple[1], 'pronoun': name_tuple[2], 'object': name_tuple[3]}
    NAME_DICTS.append(new_name)

def context_filler(string_to_fill, random_person=None, return_person=False):
    """
    Fills in a stem formatted with {name} {pronoun} and {possessive} placeholders using string formatting
    Picks a random name from above and uses it to fill in the string
    random_person is a dictionary containing the person dictionary (or is randomly picked from those above)
    Returns the context filled string and (optionally) the dictionary used for the person
    """
    random_person = random_person or random.choice(NAME_DICTS)
    if return_person:
        return string_to_fill.format(**random_person), random_person
    else:
        return string_to_fill.format(**random_person)

def make_folder_if_necessary(folder_name, path_to_folder):
    """
    This checks if path_to_folder/folder_name exists, and if it doesn't, then it creates the folder
    :param folder_name: the name of the folder to be checked, e.g. "box_plots"
    :param path_to_folder: the relative path to the folder, e.g. "../images/statistics"
    """
    full_path = "%s/%s" % (path_to_folder, folder_name)
    if os.path.isdir(full_path):
        pass
    else:
        print("Creating directory %s\n\n" % full_path)
        os.makedirs(full_path)

def itex2mml(itex_str, inner_only=False):
    """
    Converts one of our latex strings (including $_ wrappers) to MathML
    :param itex_str: string of latex wrapped in $_ wrappers
    :param inner_only: this indicates whether you only want the MathML without the outer <math></math> tags
    which is used for piecing together your own MathML, though might produce fragile MathML. User beware.
    :return: a string containing the MathML for the input

    :Example:

    >>>itex2mml("$_\frac{x+1}{5}$_")
    "<math><mfrac><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mn>5</mn></mfrac></math>"
    """

    fin, fo = os.popen2('itex2MML')
    fin.write(itex_str.replace('$_', '$'))
    fin.close()
    for line in fo:
        line = line.strip()
        if line:
            line = line.replace('&minus;', '-')
            line = line.replace('<semantics>', '')
            line = line.replace('</semantics>', '')
            line = line.replace("<math xmlns='http://www.w3.org/1998/Math/MathML' display='inline'>", "<math>")
            line = re.sub("<annotation encoding='application/x-tex'>.*?</annotation>", "", line)
            line = re.sub("<mspace .*?/>", "", line)
            line = re.sub("<mo.*?>", "<mo>", line)
            if inner_only:
                line = line.replace('<math>', '')
                line = line.replace('</math>', '')
            return line


def square_root_mml(n):
    """
    helper returns the mathml for the simplest radical form of sqrt(n)
    :param n: a positive integer
    :return: str of the mathml for sqrt(n)
    """
    square_part = largest_perfect_square(n)
    remaining_factor = n / square_part
    outer_factor = int(square_part ** (0.5))

    if outer_factor > 1:
        return "<math><mn>%d</mn><msqrt><mn>%d</mn></msqrt></math>" % (outer_factor, remaining_factor)
    else:
        return "<math><msqrt><mn>%d</mn></msqrt></math>" % remaining_factor

def fraction(numerator, denominator, dfrac=False, simplify=False):
    """
    Returns the LaTeX for the fraction without simplifying unless simplify is True
    """
    if simplify:
        return latex(sympify(numerator) / denominator)
    else:
        return "\\%sfrac{%s}{%s}" % ("d" if dfrac else "", latex(numerator), latex(denominator))

def fraction_mml(number):
    """
    Returns the MathML string for the rational number (checks for whether it's actually a fraction
    :param number: a sympy rational number
    """
    if number < 0:
        sign = "<mo>-</mo>"
    else:
        sign = ""
    if number.q == 1:
        return "<math>%s<mn>%d</mn></math>" % (sign, abs(number))
    else:
        return "<math>%s<mfrac><mn>%d</mn><mn>%d</mn></mfrac></math>" % (sign, abs(number.p), abs(number.q))

def align(*args, **kwargs):
    """
    Returns LaTeX for an aligned array of the arguments args
    args are assumed to be strings (need to be wrapped or passed to latex - will make other wrappers for things)
    Default is to do args[0] &= args[1] \\ &= args[2] \\ &= args[3] etc.

    It will eliminate any duplicate entries in args, so for example if you have an unsimplified a / b
    followed by a reduced a / b, you don't have to worry about the case where it was already simplified
    to begin with because it will just leave that line out.

    If keyword argument last_approx=True is passed, it will make the very last equality an approximation
    (probably a more robust way to do this)
    """

    args = eliminate_duplicates(args)
    if kwargs.get("last_approx", False):
        separators = [" \\\\ &= "] * (len(args) - 3)
        separators.append(" \\\\ & \\approx ")
        body_terms = zip(separators, args[2:])
        body_terms = ["".join(pair) for pair in body_terms]
        align_body = args[1] + "".join(body_terms)
    else:
        align_body = " \\\\ &= ".join(args[1:])
    align_start = "\\begin{align} %s &= " % args[0]
    align_end = " \\end{align}"
    return align_start + align_body + align_end

def align_bothsides(*args, **kwargs):
    """
    This assumes that we get both sides of each equation. Can pass "" to get an empty entry.
    """
    if kwargs.get("last_approx", False):
        separators = [" \\\\ &= "] * (len(args) - 3)
        separators.append(" \\\\ & \\approx ")
        body_terms = ""

def askip(n=6):
    """
    returns n tabs so the answer is in the right column
    Probably shouldn't use this anymore
    """
    return "\t" * n

def intersect(l1, l2):
    """
    Intersection of two lists l1 and l2
    """
    return list(set(l1) & set(l2))

def interval_notation(list_of_intervals):
    """
    Returns the latex string representation of a union of intervals.
    :param list_of_intervals: Each element is a 4-tuple (a, b, True/False, True/False)
    representing the interval (a, b) where the first True/False tells whether to include
    a and the second indicates whether to include b. Use None in the first entry to indicate -infinity
    and None in the second entry to indicate infinity
    :return: latex string representation of the union of intervals

    :Example:

    >>>interval_notation([(1, 3, True, False), (5, None, True, False)])
    "\left[1, 3 \right) \cup \left[5, \infty \right)"
    >>>interval_notation([(4, 6, True, True)])
    "\left[4, 6\right]"
    """
    def parse_interval(interval_tuple):
        """
        Helper function for parsing the individual intervals
        """
        if interval_tuple[0] == None:
            first_bracket = "\\left("
            first_entry = "-\\infty"
        else:
            first_bracket = "\\left[" if interval_tuple[2] else "\\left("
            first_entry = "%s" % latex(interval_tuple[0])
        if interval_tuple[1] == None:
            second_bracket = "\\right)"
            second_entry = "\\infty"
        else:
            second_bracket = "\\right]" if interval_tuple[3] else "\\right)"
            second_entry = "%s" % latex(interval_tuple[1])
        return "%s%s,%s%s" % (first_bracket, first_entry, second_entry, second_bracket)

    list_of_intervals = [parse_interval(interval) for interval in list_of_intervals]
    return "\\cup".join(list_of_intervals)

def tags_from_stem(question_string, leading_comma = True):
    """
    Creates a list of tags from the question string by seeing if each function is present in the string
    :param question_string: the string representing the question stem (or any string really)
    :param leading_comma: whether to include a leading comma in front of the list of tags
    :return: comma separated list of tags
    """
    d = {"\\sin":"sine",
         "\\cos":"cosine",
         "\\tan":"tangent",
         "\\sec":"secant",
         "\\log":"logarithm",
         "\\ln":"logarithm",
         "e^":"exponential"}
    list_of_tags = []
    for k in d.keys():
        if k in question_string:
            list_of_tags.append(d[k])
    if intersect(["sine","cosine","tangent","secant"], list_of_tags) != []:
        list_of_tags.append("trigonometric function")
    rs = ", ".join(list_of_tags)
    return "%s%s" % (", " if leading_comma else "", rs)

def two_row_table(x_list, y_list, x_label="x", y_label="y", use_latex=True):
    """
    Constructs the LaTeX two row table where x_list forms the first row, y_list forms the second row
    and of data given. x_label and y_label are optional labels for the rows.
    """
    number_of_entries = len(x_list)
    if not use_latex:
        table_string = "\\begin{array}{c|%s} %s & " % ("c" * (number_of_entries + 1), x_label)
        table_string += " & ".join(map(lambda x:"%g" % x, x_list))
        table_string += " \\\\ \\hline %s & " % y_label
        table_string += " & ".join(map(lambda x:"%g" % x, y_list))
        table_string += " \\end{array}"
    else:
        table_string = "\\begin{array}{c|%s} %s & " % ("c" * (number_of_entries + 1), x_label)
        table_string += " & ".join(map(lambda x:"%s" % latex(x), x_list))
        table_string += " \\\\ \\hline %s & " % y_label
        table_string += " & ".join(map(lambda x:"%s" % latex(x), y_list))
        table_string += " \\end{array}"
    return table_string

def two_col_table(x_list, y_list, x_label="x", y_label="y", is_header_text=False, use_latex=False):
    """
    Constructs the LaTeX two column table where x_list forms the first row, y_list forms the second row
    and of data given. x_label and y_label are optional labels for the rows.
    """
    if is_header_text:
        x_label = "\\text{%s}" % x_label
        y_label = "\\text{%s}" % y_label
    table_string = "\\begin{array}{c|c} %s & %s \\\\ \\hline " % (x_label, y_label)
    for x_entry, y_entry in zip(x_list, y_list):
        if use_latex:
            table_string += " %s & %s \\\\ " % (latex(x_entry), latex(y_entry))
        else:
            table_string += " %g & %g \\\\ " % (x_entry, y_entry)
    table_string += "\\end{array}"
    return table_string

def table_by_columns(*cols):
    # cols is a bunch of tuples where the first entry is the label for the column
    # and the second entry is a tuple of vales e.g.
    # table_by_columns(('x',(1,2,3)), ('y',(4,5,6)))
    labels = []
    data_columns = []
    for label, data in cols:
        labels.append(label)
        data_columns.append(data)
    label_string = " & ".join(labels)
    entry_rows = zip(*data_columns)
    entry_rows = [" & ".join(map(latex, row)) for row in entry_rows]
    table_string = "\\begin{array}{|%s}\hline %s \\\\" % ('c|'*len(labels), label_string)
    for row in entry_rows:
        table_string += " \\hline %s \\\\" % row
    table_string += "\\hline \\end{array}"
    return table_string

def preamble(concept_name,ass_name,ques_num,fr):
    return "%s\t%s\t%s\t%s\t\t%s\t" %(concept_name,"https://beta.knewton.com",ass_name,ques_num,fr)

def postamble(tags,n=7):
    #n tabs to skip over the taxonomy etc.
    return "\t"*n+"%s\t%s\t%s" %(tags,"Knewton","Standard Knewton License")

def num_ans_mults(ans,num=4,post_stem = "",mult_range=[0.2,0.25,0.5,1.5,2,3,4,5]):
    #gives a range of multipliers times the correct answer
    mults = random.sample(mult_range,num)
    answers = [ans] + [ans*m for m in mults]
    return answers_sorted(answers[0:1], answers[1:], 6, post_stem)

def sign(x):
    if x == -oo: return -1
    if x == oo: return 1
    return 0 if x==0 else x/abs(x)

def constant_sign(x, leading = False):
    """
    Gives the string x with the appropriate sign in front
    Useful for making strings involving adding a bunch of terms together
    leading tells whether the constant is first, so shouldn't have a plus sign in that case
    constant_sign(3) gives "+3"
    constant_sign(3, True) gives "3"
    constant_sign(-3) gives "-3"
    constant_sign(sympify(1) / 2) gives "+ \frac{1}{2}"
    etc.
    """
    if x >= 0 and not leading:
        return " + %s" % latex(x)
    elif x >= 0 and leading:
        return latex(x)
    elif x < 0:
        return latex(x)

def pmsign(x, leading = False):
    if leading:
        if abs(x)==1:
            return "" if x>0 else "-"
        else:
            return latex(x)
    elif sign(x) >= 0:
        return "+ %s" % (latex(x) if x != 1 else "")
    else:
        return "- %s" % (latex(abs(x)) if x != -1 else "")

def pmsign_nozero(x, var_name, leading = False):
    if leading:
        if x==1:
            return "%s" % (var_name)
        elif x==-1:
            return "- %s" % (var_name)
        else:
            return "%s %s" % (str(x), var_name)
    elif sign(x) > 0:
        return "+ %s %s" % (x if x != 1 else "", var_name)
    elif x == 0:
        return ""
    else:
        return "- %s %s" % (abs(x) if x != -1 else "", var_name)


def quad_string(a,b,c,var_sym=symbols('x')):
    # formats the string with proper plus and minus signs for ax^2 + bx + c
    return latex(a*var_sym**2 + b*var_sym + c)

def quadsolve(a,b,c):
    """
    solves the quadratic ax^2+bx+c = 0
    returns a reduced fraction tuple if it is rational
    otherwise, decimal
    """
    if is_square(b**2-4*a*c):
        return reduce_fraction(int(-b-(b**2 - 4*a*c)**(0.5)),int(2*a)),\
               reduce_fraction(int(-b+(b**2 - 4*a*c)**(0.5)),int(2*a))
    return (-b-(b**2 - 4*a*c)**(0.5))/(2*a),(-b+(b**2 - 4*a*c)**(0.5))/(2*a)

def eliminate_duplicates(l):
    new_l = []
    for i in l:
        if i not in new_l:
            new_l.append(i)
    return new_l

def rand_fracs(a,b,denom=2,num=5,do_not_include = []):
    """
    returns num random fractions with denominator equal to denom
    between a and b
    """
    if num<0: return []
    possible = [foo for foo in range(a*int(denom)+1,b*int(denom)-1) if sympify(foo)/denom not in do_not_include]
    return [sympify(foo) / denom for foo in random.sample(possible, num)]

def partition(n, length=None):
    """
    Returns a random partition of n of length length
    :param n: number being partitioned
    :param length: length of the partition
    :return: a partition of n given as a list of integers
    """
    length = length or random.randint(2, min(n - 1, 5))
    if length == 1:
        return [n]
    if n == 1:
        return [1]
    if n == length:
        return [1] * n
    new_part = random.randint(1, n / 2)
    return sorted([new_part] + partition(n - new_part, length - 1))

def is_square(n):
    """determines if n is a perfect square"""
    if n<0: return False
    return n**(.5) == int(n**(.5))

def largest_perfect_square(n):
    """returns the largest square factor of n"""
    for i in range(int(n**0.5), 0, -1):
        if n % i**2 == 0:
            return i**2
    return 1

def decimalmml(d):
    decpart = d - int(d)
    decstring = "%g" % decpart
    decstring = decstring[2:]
    intpart = int(d)
    return "<math><mn>%d</mn><mo>.</mo><mn>%s</mn></math>" % (intpart, decstring)

def reformat(p,trig=False):
    # r = p.replace("**","^")
    # r = r.replace("*","")
    # if trig:
    # 	#makes replacements of the form ()^2 to ^2()
    # 	#not super robust
    # 	l = re.findall('\(.*?\)|\^\d+', r)
    # 	for i in range(len(l)-1):
    # 		if l[i]+l[i+1] in r and l[i+1][0]=="^":
    # 			r = r.replace(l[i]+l[i+1],l[i+1]+l[i])
    # #l = re.findall('(\d*[x]\^\d)\/(\d+)',r)
    # l = re.findall('(_*[x\_n]\^\d)\/\(?(\w*[x\_n]\^\d)\)?',r)
    # for (foo,bar) in l:
    # 	if foo+'/'+bar in r:
    # 		r = r.replace(foo+'/'+bar,'\dfrac{%s}{%s}' % (foo,bar))
    # lrep = ["sin","cos","tan","sec"]
    # if trig:
    # 	for i in lrep:
    # 		r = r.replace(i,"\\%s"%i)
    # return r
    r = p.replace("\\log","\\ln")
    r = r.replace("\\operatorname{atan}","\\arctan")
    r = r.replace("\\operatorname{asin}","\\arcsin")
    r = r.replace("\\operatorname{acos}","\\arccos")
    return r


def terms_string(*args):
    """
    returns a string representing the
    list of terms added together with appropriate signs and _without simplifying_ in the given order
    this replaces the need for lots of crazy string formatting, pmsign, etc. to get signs right
    ex: terms_string(5, 3*x, -2, -x**2, -8*x) returns '5 + 3 x - 2 - x^{2} - 8 x'
    note: this will not include 0 terms
    """
    return latex(Add(*args, evaluate=False), order='none')


def terms_constants(*args):
    """
    Sum of a bunch of constants with appropriate signs, includes zeros
    """
    first_term = constant_sign(args[0], True)
    other_terms = [constant_sign(arg) for arg in args[1:]]
    return first_term + " ".join(other_terms)


def parabola_plotter(a, b, c, image_number = 0, image_label = '', image_letter = '', color = 'blue', dpi=100,
                     folder_name = 'vertical_horizontal_shifts', font_size = 20, file_name = 'vhplot', axis_size = 8,
                     points_to_plot = (), dot_size = 100):
    """
    plots the function y = a(x-b)^2 + c, saved to folder_name/file_name image_label image_number image_letter
    :param a: leading coefficient
    :param b: horizontal shift to the right
    :param c: vertical shift
    :param image_number: part of the file name that gets saved
    :param color: the color of the plot and any points plotted
    :param dpi: the dpi of the saved image
    :param folder_name: the folder into which this is saved (assumed to be one level up from where the script is)
    :param file_name: the file name (though image_label image_number and image_letter get appended
    :param image_letter: part of the file name
    :param points_to_plot: ordered pairs which are plotted with dot_size on the graph
    """
    diff = 10
    x_vals = np.linspace(b - diff, b + diff, 500)
    y_vals = a * (x_vals - b)**2 + c
    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for label in ['bottom', 'left']:
        ax.spines[label].set_position('zero') # this is what zeros the axes
        ax.spines[label].set_linewidth(4)
        ax.spines[label].set_alpha(0.6)
        ax.spines[label].set_capstyle('round')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks(non_zero_range(-axis_size + 1, axis_size))
    ax.set_yticks(non_zero_range(-axis_size + 1, axis_size))
    ax.set_xlim(-axis_size, axis_size)
    ax.set_ylim(-axis_size, axis_size)
    plt.tick_params(axis='both', labelsize=font_size, labelcolor=(0,0,0,0.6))
    plt.plot(x_vals, y_vals, c=color, linewidth= 4, alpha = 0.9)
    if points_to_plot:
        x_point_values = [point[0] for point in points_to_plot]
        y_point_values = [point[1] for point in points_to_plot]
        plt.scatter(x_point_values, y_point_values, s=dot_size, c=color)
    plt.grid(True)
    plt.savefig('../%s/%s%s%d%s.png' % (folder_name, file_name, image_label, image_number,
                                                                  image_letter), dpi=dpi, transparent=True)
    plt.cla()
    plt.close()

def function_plotter(function, image_number = 0, image_label = '', axis_size = 8, show_grid = False, dpi = 150,
                     image_letter = '', color = 'blue', folder_name = 'horizontal_line_test', file_name = 'plot',
                     font_size = 20):
    """
    Plots the function function (defined as a lambda function I think is best) on the grid -axis_size, axis_size
    """

    x_vals = np.linspace(-axis_size, axis_size, 500)
    y_vals = function(x_vals)
    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for label in ['bottom', 'left']:
        ax.spines[label].set_position('zero') # this is what zeros the axes
        ax.spines[label].set_linewidth(4)
        ax.spines[label].set_alpha(0.6)
        ax.spines[label].set_capstyle('round')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if show_grid:
        ax.set_xticks(non_zero_range(-axis_size + 1, axis_size))
        ax.set_yticks(non_zero_range(-axis_size + 1, axis_size))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlim(-axis_size, axis_size)
    ax.set_ylim(-axis_size, axis_size)
    if show_grid:
        plt.tick_params(axis='both', labelsize=font_size, labelcolor=(0,0,0,0.6))
    plt.plot(x_vals, y_vals, c=color, linewidth= 4, alpha = 0.9)
    plt.grid(True)
    plt.savefig('../%s/%s%s%d%s.png' % (folder_name, file_name, image_label, image_number,
                                                                  image_letter), dpi=dpi, transparent=True)
    plt.cla()
    plt.close()

def extreme_function_plotter(function, image_number = 0, image_label = '', x_range = (-8, 8), dpi = 150,
                     image_letter = '', color = 'blue', folder_name = 'horizontal_line_test', file_name = 'plot',
                     font_size = 20, y_tick_distance = None):
    """
    Plots the function function (defined as a lambda function I think is best) on the grid -axis_size, axis_size
    Doesn't make assumptions about the y-values, tries to use a decent scale or something
    """

    x_vals = np.linspace(x_range[0], x_range[1], 500)
    y_vals = function(x_vals)
    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for label in ['bottom', 'left']:
        ax.spines[label].set_position('zero') # this is what zeros the axes
        ax.spines[label].set_linewidth(4)
        ax.spines[label].set_alpha(0.6)
        ax.spines[label].set_capstyle('round')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # if show_grid:
    #     ax.set_xticks(non_zero_range(-axis_size + 1, axis_size))
    #     ax.set_yticks(non_zero_range(-axis_size + 1, axis_size))
    # else:
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    ax.set_xticks(range(x_range[0], x_range[1]+1))
    if y_tick_distance:
        ax.set_yticks(range(y_tick_distance, int(max(y_vals)), y_tick_distance) +
                      range(-y_tick_distance, int(min(y_vals)), -y_tick_distance))
    ax.set_xlim(x_range[0], x_range[1])
    # ax.set_ylim(-axis_size, axis_size)
    plt.tick_params(axis='both', labelsize=font_size, labelcolor=(0,0,0,0.6))
    plt.plot(x_vals, y_vals, c=color, linewidth= 4, alpha = 0.9, clip_on=False)
    plt.grid(True)
    plt.savefig('../%s/%s%s%d%s.png' % (folder_name, file_name, image_label, image_number,
                                                                  image_letter), dpi=dpi, transparent=True)
    plt.cla()
    plt.close()

def substitute_unsimplified(eval_point, poly, x=symbols('x'), include_parentheses=True, order='none'):
    """
    Gives the string you would achieve by substituting eval_point into poly without simplifying it
    :param eval_point: the value being substituted into poly
    :param poly: a polynomial
    :param x: the symbol being replaced
    :param include_parentheses: whether to put parentheses around the argument being substituted
    :return: a string of latex representing the substitution of eval_point into poly
    """
    string_of_eval = latex(eval_point)
    string_of_eval = string_of_eval.replace(" ","")
    if include_parentheses:
        eval_symbol = symbols('(%s)' % string_of_eval)
    else:
        eval_symbol = symbols(string_of_eval)
    return latex(poly.subs(x, eval_symbol), order=order)

def substitute_expression(expression_string, substitute_into, variable_to_replace, use_parentheses=True):
    """
    Does a straightforward substitution of one expression string into another
    :param expression_string: str for thing to substitute in (e.g. 'x + 1')
    :param substitute_into: str for thing to substitute into (e.g. '5x^2 + 3')
    :param variable_to_replace: str for thing in substitute_into that's being replaced e.g. x
    :param use_parentheses: whether to put parentheses around the the expression_string before substituting it in
    :return: string for the replaced version, e.g. 5(x+1)^2 + 3
    """
    if use_parentheses:
        expression_string = "(%s)" % expression_string
    return substitute_into.replace(variable_to_replace, expression_string)

def quadrant(angle, mode, as_numeral=True, with_text_wrapper=True):
    """
    Helper function gives the quadrant of an angle
    :param angle: the angle, either in degrees or radians (indicated by mode)
    :param mode: 'radians' or 'degrees'
    :param as_numeral: whether to return it as I, II, III, IV or 1, 2, 3, 4
    :return: either a string 'I', 'II', 'III', 'IV', if as_numeral, or an int 1, 2, 3, 4
    """
    if mode == 'radians':
        quadrant_int = (int(angle / (pi / 2)) % 4) + 1
    else:
        quadrant_int = (int(angle / 90) % 4) + 1

    if as_numeral:
        numeral = ["", "I", "II", "III", "IV"][quadrant_int]
        if with_text_wrapper:
            return "$_\\text{%s}$_" % numeral
        return numeral
    else:
        return quadrant_int

def serialize(*clauses):
    """
    Given foo, bar, gar returns "foo, bar, and gar"
    Given foo, bar returns "foo and bar"
    Includes oxford comma

    :param clauses: list of strings
    :return: strings properly comma'd and and-ed
    """
    if len(clauses) > 2:
        all_but_last = ", ".join(clauses[:-1])
        return "%s, and %s" % (all_but_last, clauses[-1])
    elif len(clauses) == 2:
        return "%s and %s" % (clauses[0], clauses[1])
    elif len(clauses) == 1:
        return clauses[0]
    else:
        return ''

def gcd_helper(m, n):
    """
    computes the greatest common divisor of m and n
    """
    if m<0 or n<0: return gcd(abs(m),abs(n))
    if n==0: return m
    if m<n: return gcd_helper(n,m)
    return gcd_helper(m % n, n)

def gcd(*args):
    """
    :param args: list of numbers
    :return: greatest common divisor (greatest common factor) of the list of numbers
    """
    if len(args) == 1:
        return args[0]
    return gcd_helper(args[0], gcd(*args[1:]))

def lcm(*args):
    """
    :param args: a list of numbers
    :return: the least common multiple of the list of numbers
    """
    if len(args) == 1:
        return abs(args[0])
    lcm_of_others = lcm(*args[1:])
    return abs(args[0] * lcm_of_others / gcd(args[0], lcm_of_others))

def reduce_fraction(num,denom):
    """
    gives tuple for the reduced fraction
    """
    d = gcd(num, denom)
    if denom<0: return (-num / d, -denom / d)
    else: return (num / d, denom / d)

def frac(a,b,big=False):
    """
    gives the string for the fraction a/b
    """
    try:
        # if it's a numeric fraction, reduce it
        p,q = reduce_fraction(a,b)
        #if it's actually an integer, return the integer
        if abs(q)==1: return "%d" % (p*q)
        else: return "%s\\frac{%d}{%d}" % ("\\Large" if big else "",p,q)
    except TypeError:
        return "%s\\frac{%s}{%s}" % ("\\Large" if big else "",a,b)

def tuple_to_dict(list_of_tuples):
    # given list of tuples [(a,b), (e,f), ...] returns dictionary
    # with d[a] = b, d[e] = f, etc.
    d = {}
    for (key, value) in list_of_tuples:
        d[key] = value
    return d

def random_poly(degree=3, coeff_range = 8, fixed = False, no_zeros = False, x = symbols('x')):
    """
    returns a random polynomial of degree at most degree
    fixed means it has to have degree degree
    no_zeros indicates if all coefficients should be non-zero
    """
    #n = random.randint(2,degree+1)
    f = 0
    for i in range(degree+1):
        if no_zeros:
            coeff = non_zero_select(-coeff_range,coeff_range)
        else:
            coeff = random.randint(-coeff_range,coeff_range)

            #prevents leading coefficient from being 0 if fixed is True
            while fixed and i==degree and coeff==0: coeff = random.randint(-coeff_range,coeff_range)
        f += coeff*x**i
    return f

def poly_prescribed(l, x = symbols('x')):
    """
    returns polynomial with coefficients prescribed by l, where l[0] is the constant, etc
    """
    f = 0
    d = 0
    for coeff in l:
        f += coeff*x**d
        d += 1
    return f

def coefficients(p, x = symbols('x')):
    """
    finds the coefficients of a polynomial p
    returns them as a list from the constant and upward
    """
    r = p
    l = []
    while r != 0:
        l.append(r.subs(x,0))
        r = simplify((r-r.subs(x,0))/x)
    return l

def expand_multiply(p1, p2):
    # gives the string for multiplying the polynomials p1 * p2 term by term
    c1 = coefficients(p1)
    c2 = coefficients(p2)
    list_of_coefficients = []
    # list_of_coefficieints is a list of pairs (coeff, power) for the terms in the expansion
    for i,c in enumerate(c1):
        for j,d in enumerate(c2):
            list_of_coefficients.append((c * d, i + j))
    list_of_coefficients.reverse()
    list_of_terms = []
    for i,c in enumerate(list_of_coefficients):
        if i==0:
            list_of_terms.append("%sx^{%d}" % (pmsign(c[0],True), c[1]))
        elif c[1] == 0:
            list_of_terms.append("%s %d" % ("+" if c[0]>0 else "-", abs(c[0])))
        else:
            list_of_terms.append("%sx^{%d}" % (pmsign(c[0]), c[1]))
    s = ' '.join(list_of_terms)
    s = s.replace('x^{0}','')
    s = s.replace('x^{1}','x')
    return s

def factors(n):
    # lists the factors of n, not super efficient
    i = 2
    l = [1]
    while i <= n/2:
        if n % i == 0:
            l.append(i)
        i = i + 1
    return l + [n]

def divisor_pairs(n):
    # gives pairs p,q, p<=q such that p*q = n
    if n > 0:
        facs = factors(n)
        l = []
        for i in range((len(facs) + 1) / 2):
            l.append((facs[i], n/facs[i]))
            l.append((-facs[i], -n/facs[i]))
        return l
    else:
        facs = factors(-n)
        l = []
        for i in range((len(facs) + 1) / 2):
            l.append((facs[i], n/facs[i]))
            if abs(facs[i]) != abs(n/facs[i]):
                l.append((-facs[i], -n/facs[i]))
        return l

def sum_pairs(n):
    # pairs p,q that add up to n
    return [(i,n-i) for i in range(1,n/2+1)]

def base_variable(tuple_of_strings=('x','y','t','a')):
    """
    returns a base variable [list] and symbol [list] for it
    for example, could pass the tuple ('xy', 'pq', 'ab') to get one of the pairs x y, p q, a b as variables

    :param tuple_of_strings: a tuple of strings of variables which are the options for the symbols returned
    :return: (var, var_syms) where var is the string itself (e.g. 'xy') and var_syms is the tuple of the symbols,
     e.g. (x, y)
    """
    var = random.choice(tuple_of_strings)
    char_list = [c for c in var]
    var_syms = symbols(' '.join(char_list))
    return var, var_syms

def non_zero_select(n,m=None):
    """random non zero number between -n and n or n and m (if m is specified)"""
    if n==0 and m==None: return 0
    if m==None:
        return random.choice(range(-n,0)+range(1,n+1))
    else:
        return random.choice(range(n,0)+range(1,m+1))

def non_zero_range(n,m,step=1):
    """gives range(n,m) but leaves out zero"""
    l = range(n,m,step)
    if 0 in l: l.remove(0)
    return l

def has_dupes(l):
    """checks if the list l has any duplicates
    (for making sure distractors don't match)
    """
    s = sorted(l)
    return any(s[i] == s[i+1] for i in range(len(l)-1))

def random_trig(basic = False):
    """
    returns a random trig function
    if basic, then just sine or cosine
    otherwise, tangent or secant (avoids cosecant, cotangent)
    """
    if basic: return random.choice([sin,cos])
    else: return random.choice([sin,cos,tan,sec,ln,exp])

def randomize_answers(list_of_answers, number_of_tabs=6):
    """
    randomizes a list of strings where the first is the correct answer and the rest are distractors
    e.g. randomize_answers(['foo','bar1','bar2','bar3']) could be 'bar2 foo bar1 bar3 B'
    :param list_of_answers: list of strings of answers. assume the first answer is the correct one
    :param number_of_tabs: total number of columns occupied by the answer choices (default is 6) so
    if there are fewer than 6 answer, there will be extra tabs added to pad the answers to make it take
    up 6 columns in the spreadsheet
    :return: the answer choices with
    """

    # eliminate any duplicates
    correct_answer = list_of_answers[0]
    list_of_answers = list(set(list_of_answers))
    correct_index = list_of_answers.index(correct_answer)
    list_of_answers[correct_index], list_of_answers[0] = list_of_answers[0], correct_answer

    if number_of_tabs == 0:
        number_of_tabs = len(list_of_answers)
    number_of_answers = len(list_of_answers)
    perm = random.sample(range(number_of_answers), number_of_answers)
    correct_index = perm.index(0)
    correct_letter = 'ABCDEFGH'[correct_index]
    return '\t'.join([list_of_answers[i] for i in perm]) + \
           '\t' * (number_of_tabs - number_of_answers + 1) + correct_letter

def randrange_exclude(num_els, min_num, max_num, exclude=None):
    """
    returns a list of length num_els from [min_num,max_num] inclusive
    excludes exclude if that number is given
    """
    if not exclude:
        return random.sample(range(min_num,max_num+1), num_els)
    if exclude:
        l = random.sample(range(min_num,max_num+1), num_els)
        while exclude in l:
            l = random.sample(range(min_num,max_num+1), num_els)
        return l

def randrange_exclude_list(num_els, min_num, max_num, exclude = ()):
    if not exclude:
        return random.sample(range(min_num, max_num + 1), num_els)
    l = []
    while len(l) < num_els:
        foo = random.randint(min_num, max_num)
        while foo in l or foo in exclude:
            foo = random.randint(min_num, max_num)
        l.append(foo)
    return l

def fully_formatted_question(question_stem, explanation="", answer_mathml=None, answer_choices=(),
                             correct_answer_index=None, correct_answers=(), incorrect_answers=(), number_of_answers=7,
                             multiple_answers_sorted=False, number_of_correct=None, number_of_incorrect=None):
    """
    Given a question_stem string, explanation string, list of answer_choices (if MC), or answer_mathml (if FR),
    this inserts the right number of tabs so we can finally stop having to put lots of tabs everywhere and worrying
    about things going in the wrong columns of the sheet. Will randomize the answer_choices unless a
    correct_answer_index is given, in which case they are given in order with correct_answer_index as the answer
    :param question_stem: The question stem, a string
    :param explanation: The explanation of the answer, a string
    :param answer_mathml: The answer mathml if the question is free response, a string
    :param answer_choices: The answer choices, a list of strings, if the question is multiple choice. The first
    string is assumed to be the correct one unless a correct_answer_index is given, in which case the answers
    :param correct_answer_index: the correct answer index if the answers are to be given in order (otherwise,
    first answer is assumed to be correct)
    :param correct_answers: when there are multiple correct answers these are the correct ones and get shuffled
    together with the incorrect answers
    :param incorrect_answers: when there are multiple correct answers these are the incorrect ones and get shuffled
    together with the correct answers
    :param number_of_answers: this is the number of answer columns, so helps in deciding how many to skip
    (in the case of FR questions) and how many to pad in the case of MC questions
    :return: a string with the question_stem, answer(s), explanation, appropriately tabbed
    """
    if answer_mathml != None:
        # this is a free response question
        return question_stem + "\t" * (number_of_answers + 1) + answer_mathml + "\t" + explanation
    elif answer_choices:
        # this is a multiple choice question with a single correct answer
        if correct_answer_index != None:
            if type(correct_answer_index) == type((1,)):
                # tuple of correct answers
                answer_portion = format_answers_multiple(answer_choices, correct_answer_index, number_of_answers)
            else:
                answer_portion = format_answers(answer_choices, correct_answer_index, number_of_answers)

        else:
            answer_portion = randomize_answers(answer_choices, number_of_answers)
        # print answer_portion
        # print question_stem + "\t" + answer_portion + "\t" + explanation
        return question_stem + "\t" + answer_portion + "\t" + explanation
    else:
        # must have multiple correct answers
        if multiple_answers_sorted:
            # sort the correct and incorrect answers
            all_answers = sorted(correct_answers + incorrect_answers)
            correct_indices = sorted([all_answers.index(correct_answer) for correct_answer in correct_answers])
            answer_portion = format_answers_multiple(all_answers, correct_indices, number_of_answers, True)
        elif number_of_correct:
            answer_portion = randomize_multiple(correct_answers, incorrect_answers, number_of_answers,
                                                total_correct=number_of_correct, total_incorrect=number_of_incorrect)
        else:
            answer_portion = randomize_multiple(correct_answers, incorrect_answers, number_of_answers)
        return question_stem + "\t" + answer_portion + "\t" + explanation

def format_answers(list_of_answers, correct_answer_index=0, total_number_of_answers=0):
    """
    Gives back answers with tabs between them
    Adds extra tabs if there are fewer than total_number_of_answers answers
    """
    if total_number_of_answers == 0:
        total_number_of_answers = len(list_of_answers)
    number_of_answers = len(list_of_answers)
    correct_letter = 'ABCDEFGH'[correct_answer_index]
    return '\t'.join(list_of_answers) + '\t' * (total_number_of_answers - number_of_answers + 1) + correct_letter

def format_answers_multiple(l,cor_ind = (0,), num=0, need_latex=False):
    if need_latex:
        l = ["$_%s$_" % latex(answer) for answer in l]
    if num == 0: num = len(l)
    n = len(l)
    correct_letters = ['ABCDEFGH'[answer] for answer in cor_ind]
    return '\t'.join(l) + '\t' * (num - n + 1) + ','.join(correct_letters)

def answers_sorted(correct, incorrect, num_columns=0, post_stem=""):
    """
    Given a list of correct answers and incorrect answers, return the part
    of the row of a csv which will be imported into the final spreadsheet.

    If num_columns is positive, then padding will be added to make that number
    of columns appear.

    post_stem will be appended to all answers, e.g., as a unit.

    :param list[int] correct: The list of correct answers
    :param list[int] incorrect: The list of incorrect answers
    :param int num_columns: The number of answer columns so that if the number of
        passed answers is less than this then returns blank spaces as if there were
        more.
    :param str post_stem: A suffix to append to all the answers, usually a unit.
    :return: A list that can be used as part of a row by the `csv` package.
    :rtype: list[str]
    """
    answers = sorted(correct + incorrect)
    num_columns = num_columns or len(answers)
    correct_indices = [answers.index(answer) for answer in correct]
    correct_letters = ','.join(sorted(['ABCDEF'[i] for i in correct_indices]))
    output = ["$_%s$_%s" % (latex(answer), post_stem) for answer in answers]
    output.extend([''] * (num_columns - len(output)))
    output.append(correct_letters)
    return "\t".join(output)


def randomize_multiple(correct_answers, incorrect_answers, total_spaces=6, total_correct=None, total_incorrect=None):
    """
    same as above but has multiple correct answers
    total is a tuple indicates how many distractors to take (at random)
    """
    if total_correct:
        correct_answers = random.sample(correct_answers, total_correct)
    if total_incorrect:
        incorrect_answers = random.sample(incorrect_answers, total_incorrect)
    list_of_answers = correct_answers + incorrect_answers
    number_of_answers = len(correct_answers)+len(incorrect_answers)
    answer_permutation = random.sample(range(number_of_answers), number_of_answers)
    correct_indices = [answer_permutation.index(correct_answer) for correct_answer in range(len(correct_answers))]
    correct_letters = sorted(['ABCDEFGH'[correct_index] for correct_index in correct_indices])
    correct_string = ",".join(correct_letters)
    return '\t'.join([list_of_answers[i] for i in answer_permutation]) \
           + '\t' * (total_spaces - number_of_answers + 1) + correct_string

def bunch_of_functions(correct, incorrect, total_spaces = 6):
    """
    given a bunch of functions (as strings, e.g. 3 x^2 + 5), some correct,
    some incorrect, it randomizes them and adds letters in front
    """
    list_of_answers = correct + incorrect
    n = len(correct)+len(incorrect)
    permutation_of_answer_indices = random.sample(range(n),n)
    correct_indices = [permutation_of_answer_indices.index(i) for i in range(len(correct))]
    correct_letters = sorted(['ABCDEF'[c] for c in correct_indices])
    correct_string = ",".join(correct_letters)
    list_of_answers = [list_of_answers[i] for i in permutation_of_answer_indices]
    list_of_answers = ["$_%s(x) = %s$_" % (function_letter, answer)
                       for (function_letter, answer) in zip('fghjkl', list_of_answers)]
    return '\t'.join(list_of_answers)+'\t'*(total_spaces-n+1), correct_string

def system_equations_string(coefficient_lists, b_list):
    """
    Flexible way to get the string for a system of equations
    e.g. matrix_string([[1,2],[3,4]], [5,6]) returns the system x + 2y = 5, 3x + 4y = 6
    :param coefficient_lists: a list of lists of coefficients
    :param b_list: a list of numbers representing the values on the right of the equal sign
    :return: the string for the equation array represented by the system
    """
    x, y, z = symbols('x y z')
    if len(coefficient_lists[0]) == 2:
        equation_left_sides = [a * x + b * y for (a, b) in coefficient_lists]
    else:
        equation_left_sides = [a * x + b * y + c * z for (a, b, c) in coefficient_lists]
    full_equations = ["%s &=& %d" % (latex(equation_left_side), b_value)
                      for (equation_left_side, b_value) in zip(equation_left_sides, b_list)]
    system_string = "\\left \\{ \\begin{array}{rl} %s \\end{array}\\right." % (" \\\\ ".join(full_equations))
    return system_string

def equation_add(coefficient_lists, b_list, equation_numbers, multiplication_numbers):
    """
    The string for adding a pair of equations
    e.g. equation_add([[1,2,3],[4,5,6]], [5,6]) returns the system x + 2y + 3z = 5, 4x + 5y + 6z = 6
    atop one another with a line for
    :param coefficient_lists: a list of lists of coefficients
    :param b_list: a list of numbers representing the values on the right of the equal sign
    :return: the string for the equation array represented by the system
    """
    x, y, z = symbols('x y z')
    equation_left_sides = [a * x + b * y + c * z for (a, b, c) in coefficient_lists]
    full_equations = ["%s &=& %d & \\text{(%d) multiplied by $%d$}" %
                      (latex(equation_left_side), b_value, equation_number, multiplication_number)
                      for (equation_left_side, b_value, equation_number, multiplication_number)
                      in zip(equation_left_sides, b_list, equation_numbers, multiplication_numbers)]
    system_string = "\\begin{array}{rll} %s \\hline  \\end{array}\\right." % (" \\\\ ".join(full_equations))
    return system_string