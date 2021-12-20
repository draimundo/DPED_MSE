from typing import List
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import re
import numpy as np

res_dir = "C:/Users/draimundo/Desktop/learningrates/"
loss_plot = ['metric_psnr']


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


files = [f for f in os.listdir(res_dir) if os.path.isfile(res_dir + f)]
loss_dic = {}

for file in tqdm(files):
    with open(os.path.abspath(res_dir+file)) as f:
        f_dic = {}
        lines = f.readlines()

        for line in lines:
            strings = re.findall("[a-z_]+", line)
            numbers = re.findall(" [0-9.]+", line)
            numbers = [float(i) for i in numbers]

            strings.remove('step')
            iter = int(numbers.pop(0))

            f_dic[-1] = strings
            f_dic[iter] = numbers

        loss_dic[file.split(".")[0]] = f_dic

for method in tqdm(loss_dic.keys()):
    print(method)
    for loss in loss_plot:
        try:
            index = loss_dic[method][-1].index(loss)
        except ValueError:
            break
        iters = list(loss_dic[method].keys())
        iters.remove(-1)
        
        print(np.argmin([loss_dic[method][i][index] for i in iters])) # if i in range(0,301000)

        values = [loss_dic[method][i][index] for i in iters]
        plt.plot(iters, smooth(values,0.9), label=method+'->'+loss)

plt.legend(loc="lower right")
plt.yscale('log')
plt.xlim(left=0)
plt.xlabel("Epoch")
plt.ylabel('-'.join(loss_plot))
plt.title((res_dir.split('/')[-2]))
# plt.ylim(bottom=0.6, top=8)
plt.grid(True)
plt.show()