# LearningChain

## Code Catalog
### [data](data)     ---Store the five datasets involved in the article
###### （Due to the current review status of the paper, we have only released a portion of the fifth dataset）
###### [BPD1.csv](data%2FBPD1.csv)
###### [BPD2.csv](data%2FBPD2.csv)
###### [BPD3.csv](data%2FBPD3.csv)
###### .......
### [ensemble](ensemble)   ---Ensemble-Learning code and experimental results

[light_ensemble.py](ensemble%2Flight_ensemble.py) It is the code for the BPD dataset

[light_ensemble_5.py](ensemble%2Flight_ensemble_5.py) It is the code for the HFBTP dataset

### [Meta](Meta)   ---Meta-Learning code and experimental results


[MetaReg.py](Meta%2FMetaReg.py)

## Operation mode

###### （Please do not change the relative path between the dataset folder and the code folder after downloading the git file）

#### Ensemble-Learning

````
# dataset: Used to specify which dataset to use 
# task: Used to specify which task to run

python light_ensemble.py --batch_size=10  --learning_rate=0.001 --dataset=1 --task=1
````

#### Meta-Learning

````
# targetdatasettargetdataset: Used to specify which dataset to use 
# task: Used to specify which task to run
# test_batch_size: Used to specify how many samples to use for training in the target task

python MetaReg.py --targetdataset=5 --task=1  --test_batch_size=10
````
