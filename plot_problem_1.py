import matplotlib.pyplot as plt
import numpy as np

sizes = np.array([1, 10, 100, 1000])  # MB
gpus = [2, 4, 8]

# time in ms for each configuration
measured = {
    2: [1.97, 2.11, 11.90, 92.79],
    4: [6.76, 7.42, 20.45, 150.51],
    8: [14.77, 14.48, 34.11, 227.25]
}

theoretical = [ 1000 * (12 * n) / (2 * (n - 1)) for n in gpus ]
colors = ['r', 'g', 'b']

for n in gpus:
    times = np.array(measured[n]) / 1000  # convert ms to s
    throughputs = sizes / times  # MB/s
    measured[n] = throughputs.tolist()

plt.figure(figsize=(8,6))
for n in gpus:
    plt.plot(sizes, measured[n], marker='o', label=f"{n} GPUs (measured)", color=colors[gpus.index(n)])
plt.hlines(theoretical, colors=['r', 'g', 'b'], linestyles='dashed', 
           xmin=sizes[0], xmax=sizes[-1])


plt.xscale('log')
plt.xlabel("Message size (MB)")
plt.ylabel("Throughput (MB/s)")
plt.title("All-Reduce Throughput: Distributed Communication Single Node")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("problem_1_plot.png")
plt.show()
