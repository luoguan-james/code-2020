#-*-coding:gbk-*-

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

wuhan = pd.read_csv("数据集/2019-nCoV-bd-general-20200223.csv")

day = np.arange(1,32,1)

confirmed = wuhan["confirmed"][397:428]

plt.plot(day,confirmed)

plt.scatter(day,confirmed,color="r")

plt.title("福建省近三十二天确诊人数变化曲线")

plt.show()
