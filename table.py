# -*- coding: utf-8 -*-

import numpy as np
import sympy as sym


"""
Created on Mon Jul 18 22:45:44 2016

@author: ketchers
"""

class Table(object):
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
    
    def __init__(self, *entries, **kwargs):
        self.__dict__.update(kwargs) # This add fields for each kwarg
        self.col_headers = getattr(self, 'col_headers', None)
        self.row_headers = getattr(self, 'row_headers', None)
        self.container = getattr(self, 'container', True)
        self.data = self.validate(entries)
     
    

    def validate(self, entries):
        """
        Do some checks to make sure the entreis are of the correct type.
        
        entries should be a list of lists, an array, ...
        """


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
                raise ValueError("The data is ill formed.")

        return data
        

    def build_rows_html(self):
        """
        This builds and returns the actual table.
        """
        
        data = self.data
        col_headers = self.col_headers
        row_headers = self.row_headers
        
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

                     
            self.col_headers = col_headers
            self.row_headers = row_headers
        
            out_string += "\n\t<div class=\'row-rk head-rk\'>"
                
            for h in col_headers:
                out_string += "\n\t<div class=\'cell-rk\'>%s</div>" % h
            
            out_string += "\n</div>\n"

        for j in range(len(data)):
        
            out_string += "<div class=\'row-rk\'>" 
            
            if row_headers is not None:
                out_string += "\n\t<div class=\'cell-rk head-rk\'>" + \
                    str(row_headers[j]) + "</div>" + \
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
                    

    def build_rows_latex(self):

        data = self.data
        col_headers = self.col_headers
        row_headers = self.row_headers
        
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
                    
            self.col_headers = col_headers
            self.row_headers = row_headers
           
            out_string += '&'.join(["\\textbf{%s}" % x for x in col_headers])
            out_string += "\\\\ \\hline \n"
            
        for j in range(len(data)):   

            if row_headers is not None:
                out_string += "\\textbf{%s} & " % row_headers[j] + \
                    "&".join([("%.3g" % i if str(type(i)).find('float') != -1\
                    else "\\text{%s}" % i) for i in data[j]])
                out_string += "\\\\ \\hline \n"
            else:
                out_string += "&".join([("%.3g" % i if str(type(i))\
                .find('float') != -1 else "\\text{%s}" % i)\
                     for i in data[j]])
                out_string += "\\\\ \\hline \n"

        return out_string
        
    @staticmethod
    def get_style():
        style = """
        <style type="text/css">

            .outer-container-rk {
                display: block;
                overflow-x: auto; 
            }


            .par {
                margin: 10px;
                padding: 5px;
            }

            .centering-rk {
                display: table;
                margin: auto;
                text-align: center;
                border-width: 2pt;
                border-style: solid;
                border-color: rgba(200,100,200,.5);
                border-radius: 10px;
                padding: 5px;

            }

            .container-rk {
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



            .tbl-rk {
                display: table;
            }

            .row-rk {
                display: table-row;
                padding: 5px;
            }

            .row-rk:nth-child(odd) {
                background-color: rgba(200, 200, 200, .5);
            }

            .head-rk {
                display: table-header-group;
                font-weight: bold;
            }

            .cell-rk {
                display: table-cell;
                text-align: left;
                vertical-align: middle;
                padding: 5px;
            }

            .rk {
                margin: 5px;
            }
        </style>

        """ 
        return style

    def html(self):
        
        if self.container:
            tbl = """
                  <div class=\'outer-container-rk\'>
                  <div class=\'centering-rk\'>
                  <div class=\'tbl-rk\'>
                  """ 
        else:
            tbl = "<div class=\'tbl-rk\'>" 


        tbl += self.build_rows_html()

        if self.container:
            tbl += "\n\t\t</div>\n\t</div>\n</div>\n<br>\n"
        else:
            tbl += "\n</div>\n<br>\n"
        return tbl

    def latex(self):
        
        if self.row_headers is not None:
            n_cols = 1 + len(self.data)
        else:
            n_cols = len(self.data)
            
        tbl = "\\begin{array}{|%s}\\hline\n" % ('c|' * n_cols)
        tbl += self.build_rows_latex()
        tbl += "\\end{array}"
        return tbl
        
if __name__ == "__main__":
    
    # Data can be entered as a table
    tb = Table(np.arange(20).reshape(4,5), 
               col_headers = ['Silly','a','b','c','d','e'],
               row_headers=['A','B','C','D'])
    
    # Data can be intered in as varargs "rows"
    tb1 = Table([1,2,3],[4,5,6],[7,8,9],
                     col_headers=["col1","col2",'col3'],
                     row_headers=['row1','row2','row3'])
        
    style = Table.get_style()  

    print(style)
                              
    print(tb.html())
    print(tb.latex())
        
    print(tb1.html())
    print(tb1.latex())