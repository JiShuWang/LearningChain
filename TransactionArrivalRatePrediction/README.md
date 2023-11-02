# LearningChain
This repository contains the author's implementation in Python of **prediction of arrival rate of transaction** for the paper "LearningChain: A Highly Scalable and Applicable Learning-Based Blockchain Performance Optimization Framework".
## Dependencies
- Python
- numpy
- pandas
- statsmodels
- matplotlib
- seaborn
- torch
- scikit-learn
## Implementation
Here, we provide an implementation of the **prediction of arrival rate of transaction** in **LearningChain**. The repository is organized as follows:
- **The prediction of arrival rates of transaction**

  - `data/` contains transaction send rates dataset (1min, 2mins, 5mins);

  - `model/dataset.py` contains procedures for data preprocessing for transaction send rates dataset;
  
  - `model/ARIMA.py` contains the implementation of ARIMA;

  - `model/RNN.py` contains the implementation of LSTM;
  
  - `model/TCN.py` contains the implementation of TCN;

  - `model/train.py` puts all of the above together and may be used to execute a full training run on transaction send rates dataset.