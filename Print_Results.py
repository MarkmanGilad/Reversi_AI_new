import numpy as np
import torch
import matplotlib.pyplot as plt

Directory = 'Data'
results_path = ['results_fix_12.pth', 'results_fix_13.pth']
random_results_path = ['random_fix_12.pth', 'random_fix_13.pth']

results = []
for path in results_path:
    results.append(torch.load(Directory+'/'+path))

random_results = []
for path in random_results_path:
    random_results.append(torch.load(Directory+'/'+path))

for i in range(len(results)):
    print(results_path[i], max(results[i]['results']), np.argmax(results[i]['results']), len(results[i]['results']))

with torch.no_grad():
    for i in range(len(results)):
        fig, ax_list = plt.subplots(3,1)
        fig.suptitle(results_path[i])
        ax_list[0].plot(results[i]['results'])
        ax_list[1].plot(random_results[i])
        ax_list[2].plot(results[i]['avglosses']) 

plt.show()