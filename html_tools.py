# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:33:51 2016

@author: ketchers
"""

import numpy as np
import random

def make_table(col_headers = None, row_headers = None, *entries ):
    """
    This should make a well formatted html table.
    
    Parameters:
    ----------
    col_headers : (Required) list or None
        This should be a list of headers that go across the top of the table.
    row_headers : (Required) list or None
        This should be a list of headers for rows of the table. 
    entries : (Required) list[list], np.array, or var args style (list, list, list, ....) 
        This should fill out the main body of the table and can be alist of rows,
        an array or of the form "list1, list2, ...." each a row. 
    
    Entries could be lists
    """
    
    
    def build_rows(data, row_headers = None, col_headers = None):
        """
        data: list[list]
            It is assumed that the data is in this format
        row_header: list
            If non-empty, it is assumed that len(row_header) = len(data)
        """
        out_string = ""
        
        row_len = len(data[0])
        
        if col_headers is not None:
            
            if row_headers is not None and len(col_headers) == row_len:
                if len(row_headers) == len(data) + 1:
                    col_headers = [row_headers.pop(0)] + col_headers
                elif len(row_headers) == len(data):
                    col_headers = [""] + col_headers
                else:
                    raise ValueError("len(row_headers) should be #rows or #rows + 1")
                
                
            out_string += "<tr class=\'%s\'>" % t_class
            for h in col_headers:
                out_string += "\n<th class=\'%s\'>%s</th>" % (t_class, h)
            out_string += "</tr>"
        
        for j in range(len(data)):
            out_string += "<tr class=\'%s\'>\n" % t_class
            if row_headers is not None:
                out_string += "<th class=\'%s\'>" % t_class + str(row_headers[j]) + "</th>\n" + \
                "".join(["<td class=\'%s\'>" % t_class + \
                         ("%.3f" % i if str(type(i)).find('float') != -1 else "%s" % i) +\
                         "</td>\n" for i in data[j]])
            else:
                out_string += "".join(["<td class=\'%s\'>" % t_class + \
                                       ("%.3f" % i if str(type(i)).find('float') != -1 else "%s" % i) + "</td>\n" 
                                       for i in data[j]])
            out_string += "</tr>\n"
        return out_string

    # do some checks to make sure the entreis are of the correct type
    # entries is a list the items should be one of: (1) lists (all same
    # length) these will be the rows or (2) len(entries) = 1 and entries[0]
    # is a np.array or a list or lists (all same length)

    def validate(entries):

        if len(entries) > 1 and \
            all(map(lambda x: type(x) in [list, type(np.array([]))], entries)) and \
            all(map(lambda x: len(x) == len(entries[0]), entries)):
            data = list(entries)
        elif len(entries) > 1:
            data = [list(entries)]

        if len(entries) == 1:
            if type(entries[0]) is type(np.array([])):
                data = list(entries[0])
                print("First entry is array")
            elif all(map(lambda x: type(x) is list, entries[0])) and \
                        all(map(lambda x: len(x) == len(entries[0][0]), entries[0])):
                data = entries[0]
                print("First entry is list of lists")
            elif type(entries[0]) is list:
                data = list(entries)
                print("First entry is a single list.")
            else:
                raise VaueError("The data is ill formed.")

        return data
    
    
    t_class = "tbl" + str(random.randint(1000, 2000))
    tbl = """
    <style>
        .%s {
            padding: 10px;
            text-align: left;
            border: 0px;
            font-family: sans-serif, serif;
        }
        
        table.%s {
            border-collapse: collapse;
            margin-left: auto;
            margin-right: auto;
        }
        
        div#tblcontainer {
            display: inline-block;
            padding: 5px;
            border-radius: 10px;
            border-style: solid;
            border-width: 1px;
            border-color: rgba(170, 100, 170, .5);
            margin-left: auto;
            margin-right: auto;
        }
        
        div.%s {
            display: block;
            padding: 5px;
            margin-left: 10%%;
            margin-right: 10%%;
            overflow-x: auto;
        }
        
        th.%s {
            font-weight: bold;
            text-align: center;
        }
        
        td.%s {
            text-align: center;
        }
        
        tr.%s:nth-child(odd) {
            background-color: rgba(170, 170, 200, .5);
        }
        
        tr.%s {}
    </style>
    <div class=\'%s\'>
     <div id='tblcontainer'>

    <table class=\'%s\'>
    """ % tuple([t_class] * 9)
    
    data = validate(entries) 
     
    tbl += build_rows(data, row_headers, col_headers)
        
    tbl += "</tr>\n</table>\n</div>\n</div><br>"
    return tbl


def html_image(image_url, height = 40, width = 40, preview = True):
    """
    Insert an image into a problem. If preview is set to "True", then this
    sets it up for previewing by inserting "<img src="image_url ....>".
    If preview is "False", then this inserts ${image_url}$"
    """
    if preview:
        ret = "<img src=\'%s\'%s%s>" \
        %(image_url, 
          ' width=%s ' % width if width is not None else "", 
          ' height=%s ' % height if height is not None else "")
    else:
        ret = "${%s}$"%image_url
    return ret
        
    
if __name__ == "__main__":
    e = [1.2345,2.45234,2.1428]
    o = [1,2,3]
    cl = ['Day 1', 'Day 2', 'Day 3']
    cl_alt = ['Days'] + cl
    rl = ['Expected', 'Observed']
    rl_alt = ['Days'] + rl
    print(make_table(None, None, e, o))
    print(make_table(cl, rl, e, o))
    print(make_table(cl_alt, rl, e, o))
    print(make_table(cl, rl_alt, e, o))