import json
from tqdm import tqdm
file_train = json.load(open("../dataset/temp/labels/AgriculturalDisease_train_annotations.json","r",encoding="utf-8"))
dist = {}
for file in tqdm(file_train):
    ids = file["disease_class"]
    if ids not in dist.keys():
        dist[ids] = 1
    else:
        dist[ids] += 1
print(dist)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
color = ['red', 'black', 'brown', 'green']
for i, key in enumerate(sorted(dist)):
   plt.bar(key, dist[key], color = color[np.random.randint(0,4)], width=0.4) 
   #plt.text(i+0.05,dist[key]+0.05,'%.2f' % dist[key], ha='center',va='bottom')
plt.xticks(np.arange(len(dist)), sorted(dist.keys()))
#plt.yticks(sorted(dist.values()))
#plt.grid(True)
plt.ylabel('num')
plt.legend(loc='best')
plt.xlabel("class")
plt.show()

