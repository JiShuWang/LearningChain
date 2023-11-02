import numpy as np
import pandas as pd
import torch

from dataset import LoadData

test_data = LoadData(data_path='../data/transaction_send_rates/ReserveSendRates_2minutes.npy', divide_days=[24, 6],
                     time_interval=2, history_length=12, train_mode='test')

y_true = torch.tensor(np.load("../result/y_true_arima_2mins.npy").reshape(4320, )).reshape(4320, 1, 1)
y_pred = torch.tensor(np.load("../result/y_pred_arima_2mins.npy").reshape(4320, )).reshape(4320, 1, 1)

y_true = test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], y_true).numpy().reshape(4320, )
y_pred = test_data.recover_data(test_data.park_norm[0], test_data.park_norm[1], y_pred).numpy().reshape(4320, )

y_true = pd.Series(y_true, name="y_true")
y_pred = pd.Series(y_pred, name="y_pred")

y = pd.concat([y_true, y_pred], axis=1)
y.to_csv("../result/y_true_pred_arima_2mins.csv")
