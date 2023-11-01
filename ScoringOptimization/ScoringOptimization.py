import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The features of data:
# Transaction Arrival Rate, Block Size, Orderer, pre_Latency, pre_Throughput

data = pd.read_csv("../Data/pre.csv")  # The file that predicted blockchain performance data
data = np.array(data)
data = data.tolist()

Throughput = {}  # The results of throughput score
Latency = {}  # The results of latency score

BlockSize = []  # The list of all can be selected block size
for blocksize in range(10, 805, 5):
    BlockSize.append(str(blocksize))
BlockSize.append("best")

for blocksize in BlockSize:
    globals()["Score_"+blocksize] = [] # Using dynamic variable names to store scores and performance corresponding to each block size
globals()["Score_best"] = {}

for content in data:
    transactionarrivalrate = content[0] # as the key of dict
    blocksize = content[1]
    orderer = content[2]
    pre_latency = content[3]
    pre_throughput = content[4]
    Throughput.setdefault(transactionarrivalrate, [9999, 0])  # Throughput_max, Throughput_min
    Latency.setdefault(transactionarrivalrate, [9999, 0])  # Latency_max, Latency_min
    Throughput[transactionarrivalrate][0] = min(Throughput[transactionarrivalrate][0], pre_throughput)  # Throughput_min
    Throughput[transactionarrivalrate][1] = max(Throughput[transactionarrivalrate][1], pre_throughput)  # Throughput_max
    Latency[transactionarrivalrate][0] = min(Latency[transactionarrivalrate][0], pre_latency)  # Latency_min
    Latency[transactionarrivalrate][1] = max(Latency[transactionarrivalrate][1], pre_latency)  # Latency_max

    globals()["Score_best"].setdefault(transactionarrivalrate,[0, 0, 0])
    if orderer == 3: # Only select data that the number of orderers equals 3
        globals()["Score_"+str(int(blocksize))].append([transactionarrivalrate,pre_latency,pre_throughput])

Weight_Lat, Weight_Thr = 0, 1  # Throughput is main factor
# Weight_Lat, Weight_Thr = 1, 0  # Latency is main factor

for content in data:  # Min-Max
    transactionarrivalrate = content[0]  # as the key of dict
    blocksize = content[1]
    orderer = content[2]
    pre_latency = content[3]
    pre_throughput = content[4]
    score_Latency = (pre_latency - Latency[transactionarrivalrate][1]) / (
            Latency[transactionarrivalrate][0] - Latency[transactionarrivalrate][1])
    score_Throughput = (pre_throughput - Throughput[transactionarrivalrate][0]) / (
            Throughput[transactionarrivalrate][1] - Throughput[transactionarrivalrate][0])
    score = Weight_Lat * score_Latency + Weight_Thr * score_Throughput
    if score > globals()["Score_best"][transactionarrivalrate][0]:
        globals()["Score_best"][transactionarrivalrate] = [score, blocksize, orderer, pre_latency, pre_throughput]

print("Avg Throughput")
for blocksize in BlockSize:
    if blocksize != "best":
        print("Block Size: "+blocksize+", "+str(sum([i[2] for i in globals()["Score_"+blocksize]]) / len(globals()["Score_"+blocksize])))
    else:
        print("Block Size: " + blocksize + ", " + str(sum([globals()["Score_"+blocksize][i][4] for i in globals()["Score_"+blocksize]]) / len(globals()["Score_"+blocksize])))

print("Avg Latency")
for blocksize in BlockSize:
    if blocksize != "best":
        print("Block Size: "+blocksize+", "+str(sum([i[1] for i in globals()["Score_"+blocksize]]) / len(globals()["Score_"+blocksize])))
    else:
        print("Block Size: " + blocksize + ", " + str(sum([globals()["Score_" + blocksize][i][3] for i in globals()["Score_" + blocksize]]) / len(globals()["Score_" + blocksize])))
