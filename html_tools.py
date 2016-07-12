# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 21:33:51 2016

@author: ketchers
"""

import numpy as np
import random


# This is a place wher pytho 3 would shine, the way python2 handles varargs
# is not nice.
def make_table(col_headers = None, row_headers = None, 
               container=True, *entries ):
    """
    This should make a well formatted html table. (This is pure CSS3)
    
    Parameters:
    ----------
    col_headers : (Required) list or None
        This should be a list of headers that go across the top of the table.
    row_headers : (Required) list or None
        This should be a list of headers for rows of the table. 
    container   : Boolean
        If true this wraps the table in a container with a border that adds
        a scrollbar at the bottom if window is small. (Responsive)
    entries     : (Required) list[list], np.array, 
                  or var args style (list, list, list, ....) 
        This should fill out the main body of the table and can be alist of rows,
        an array or of the form "list1, list2, ...." each a row. 
    
    Entries could be lists
    """
    
    
    def build_rows(data, row_headers = None, col_headers = None, 
                   latex = False):
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
                
                
            out_string += "\n\t<div class=\'row-rk head-rk\'>"
            for h in col_headers:
                out_string += "\n\t<div class=\'cell-rk\'>%s</div>" % h
            out_string += "\n</div>\n"
        
        for j in range(len(data)):
            out_string += "<div class=\'row-rk\'>" 
            if row_headers is not None:
                out_string += "\n\t<div class=\'cell-rk head-rk\'>" \
                    + str(row_headers[j]) + "</div>" + \
                "".join(["\n\t<div class=\'cell-rk\'>" + \
            ("%.3g" % i if str(type(i)).find('float') != -1 else "%s" % i) +\
                         "</div>" for i in data[j]])
            else:
                out_string += "".join(["\n\t<div class=\'cell-rk\'>" + \
            ("%.3g" % i if str(type(i)).find('float') != -1 else "%s" % i) + \
                                       "</div>" 
                                       for i in data[j]])
            out_string += "\n</div>\n"
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
            if type(entries[0]) is type(np.array([])) and\
               type(entries[0][0]) in [list, type(np.array([]))]:
                data = list(entries[0])
            elif type(entries[0]) is type(np.array([])):
                data = entries
            elif all(map(lambda x: type(x) is list, entries[0])) and \
                        all(map(lambda x: len(x) == len(entries[0][0]), entries[0])):
                data = entries[0]
            elif type(entries[0]) is list:
                data = entries
            else:
                raise VaueError("The data is ill formed.")
        
        return data
        
    style = """
    <style type="text/css">

        div.outer-container-rk {
            display: block;
            overflow-x: auto; 
        }


        div.par {
            margin: 10px;
            padding: 5px;
        }
        
        div.centering-rk {
            display: table;
            margin: auto;
            text-align: center;
            border-width: 2pt;
            border-style: solid;
            border-color: rgba(200,100,200,.5);
            border-radius: 10px;
            padding: 5px;

        }

        div.container-rk {
            /*float: left;*/
            display: inline-block;
            /*            
            border-width: 2pt;
            border-style: solid;
            border-color: rgba(200,100,200,.5);
            border-radius: 10px;
            padding: 5px;
            */
        }

       

        div.tbl-rk {
            display: table;
        }

        div.row-rk {
            display: table-row;
            padding: 5px;
        }

        div.row-rk:nth-child(odd) {
            background-color: rgba(200, 200, 200, .5);
        }

        div.head-rk {
            display: table-header-group;
            font-weight: bold;
        }

        div.cell-rk {
            display: table-cell;
            text-align: left;
            vertical-align: middle;
            padding: 5px;
        }

        figure.rk {
            margin: 5px;
        }
    </style>
   
    """ 
    
    if container:
        tbl = """
        <div class=\'outer-container-rk\'>
            <div class=\'centering-rk\'>
                <div class=\'tbl-rk\'>
        """ 
    else:
        tbl = "<div class=\'tbl-rk\'>" 
    
    data = validate(entries) 
     
    tbl_string = build_rows(data, row_headers, col_headers)
        
    tbl += tbl_string
    
    if container:
        tbl += "\n\t\t</div>\n\t</div>\n</div>\n<br>\n"
    else:
        tbl += "\n</div>\n<br>\n"
    return tbl, style
    
    
def html_image(image_url, height = None, width = None, 
               display = None, preview = True):
    """
    Insert an image into a problem. If preview is set to "True", then this
    sets it up for previewing by inserting "<img src="image_url ....>".
    If preview is "False", then this inserts ${image_url}$"
    """
    if preview:
        ret = """
        <div class='img'>
        <img src=\'%s\'%s%s%s>
        </div>
        """ \
        %(image_url, 
          ' width=%s ' % width if width is not None else "", 
          ' height=%s ' % height if height is not None else "",
          ' display=\'%s\' ' % display if display is not None else "")
    else:
        ret = "${%s}$"%image_url
    return ret

    
if __name__ == "__main__":
    e = [1.2345,2.45234,2.1428,1.2345,2.45234,2.1428,1.2345,2.45234,2.1428]
    o = range(1,len(e) + 1)
    cl = map(lambda x: 'Day' + str(x), o)
    cl_alt = ['Days'] + cl
    rl = ['Expected', 'Observed']
    rl_alt = ['Days'] + rl
    tbl1, style = make_table(None, None, True, e, o)
    tbl2, _ = make_table(cl, rl, True, e, o)
    tbl3, _ = make_table(cl_alt, rl, True, e, o)
    tbl4, _ = make_table(cl, rl_alt, True, e, o)
    tbl5, _ = make_table(cl, rl_alt, True, np.array([e, o]))
    tbl6, _ = make_table(cl, None, True, np.array(e))
    ex = style + tbl1 + tbl2 + tbl3 + tbl4 + tbl5 + tbl6
    print(ex)