# -*- coding: utf-8 -*-
"""
@Time : 2022/1/6 21:32 
@Author : Chao Zhu
"""
import numpy as np
import pandas as pd


y_true = pd.Series(np.load('../result/y_true_arima_1min_recover.npy').reshape(8640, ), name='y_true')
y_pred = pd.Series(np.load('../result/y_pred_arima_1min_recover.npy').reshape(8640, ), name='y_pred')

y = pd.concat([y_true, y_pred], axis=1)
y.to_csv('../result/y_true_pred_arima_1min_recover.csv')
print(y)
