import pandas as pd
import numpy as np
import glob
import os

x = glob.glob(r"dataset/train/America/*.*")

num=0
for filename in x:
    num=num+1
    ext=filename.split('\\')[1].split('.')[-1]
    name=f'America_A_{num}'
    path=filename.split('\\')[0]
    new_path=path+'\\'+name+'.'+ext
    os.rename(filename, new_path)
