#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:52:05 2019

@author: xander999
"""

import pandas as pd

data = pd.read_csv('letter-recognition.data', sep=",", header=None)
data.columns = ["letter-name","x-box","y-box","width-box","height-box","onpix","xbar-mean","ybar-mean","x-variance","y-variance","x-y-correlation","x*x*y-mean","x*y*y-mean","xEdge-mean","x-edge-Corr","yEdge-mean","y-edge-Corr"]

data.to_csv("/home/xander999/PROJECTS/SOFT_COMPUTING/Project1/data1.csv")

