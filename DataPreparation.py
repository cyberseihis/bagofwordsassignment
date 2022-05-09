import re
import numpy as np

def bagFromLine(line):
    l = line.rstrip();
    l = re.sub(r"<\d+> ", '', l)
    l = [float(x) for x in l.split()]
    l = np.bincount(l, minlength=8520)
    return l

def inputsFromFile(path):
    filer = open(path,'r')
    bags = map(bagFromLine, filer)
    return np.array(list(bags))

def labelFromLine(line):
    l = [int(x) for x in l.rstrip().split()]

def labelsFromFile(path):
    filer = open(path,'r')
    labels = [[float(x) for x in l.rstrip().split()] for l in filer]
    return np.array(labels)
