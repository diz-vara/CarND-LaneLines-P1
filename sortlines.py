# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:29:35 2017

@author: diz
"""

def sortlines(lines, top, bottom):
    
    for _line in lines:
        line = _line[0]
        x0 = line[0]
        x1 = line[2]
        w = x1 - x0
        h = line[3] - line[1]
        
        if w > 0:
            x0 = x1 + (bottom-line[3]) * w /h
            x1 = x1 + (top-line[3])*w / h
        #if (x0 < 0 or x1 < 0 or x0 > 480 or x1 > 480):
        #    line = None
        line[0] = x0
        line[1] = bottom
        line[2] = x1
        line[3] = top
    
    out = [l for l in lines if (l[0][0] > 0 and l[0][2] > 0 and l[0][0] < 480 and l[0][2] < 480)   ]                       
    return out
    

