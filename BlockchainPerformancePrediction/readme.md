# LearningChain Core Code

## Code Catalog
### [data](data)     ---Stores the five datasets involved in the article
###### [BPD1.csv](data%2FBPD1.csv)
###### [BPD2.csv](data%2FBPD2.csv)
###### [BPD3.csv](data%2FBPD3.csv)
###### .......
### [ensemble](ensemble)   ---ensemble learning code and experimental results

[light_ensemble.py](ensemble%2Flight_ensemble.py) is the code for the first to fourth datasets

[light_ensemble_5.py](ensemble%2Flight_ensemble_5.py) s the code for the 5th dataset

### [Meta](Meta)   ---Meta learning code and experimental results


[MetaReg.py](Meta%2FMetaReg.py)

## Operation mode

###### （Do not change the relative path between the dataset folder and the code folder after downloading the git file）

#### Ensemble Learning

````
# Dataset is used to specify which dataset to use

# Task is used to specify the number of tasks to run

python light_ensemble.py --batch_size=10  --learning_rate=0.001 --dataset=1 --task=1
````

#### Meta Learning

````
# targetdatasettargetdataset, dataset is used to specify which dataset to use
# task, is used to specify the number of tasks to run
# test_batch_size, is used to specify how many samples to use for training in the target task

python MetaReg.py --targetdataset=5 --task=1  --test_batch_size=10
````
