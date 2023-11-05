import matplotlib.pyplot as plt
import numpy as np

with open('flops_mat.txt', 'r') as infile:
    lines = infile.readlines()

# the matrix is transposed
flops_mat = np.array([[float(value) for value in line.split()] for line in lines]).T

with open('time_mat.txt', 'r') as infile:
    lines = infile.readlines()

time_mat = np.array([[float(value) for value in line.split()] for line in lines]).T

n_list = [2**i for i in range(25)]

plt.figure()
plt.title("Operations per time")
plt.plot(n_list,flops_mat[0], label = "f1")
plt.plot(n_list,flops_mat[1], label = "f1_vec")
plt.plot(n_list,flops_mat[2], label = "f2")
plt.plot(n_list,flops_mat[3], label = "f2_vec")
plt.xscale('log')
plt.xlabel("N")
plt.ylabel("GFlops/s")
plt.legend()

plt.figure()
plt.title("Runtime")
plt.plot(n_list,time_mat[0], label = "f1")
plt.plot(n_list,time_mat[1], label = "f1_vec")
plt.plot(n_list,time_mat[2], label = "f2")
plt.plot(n_list,time_mat[3], label = "f2_vec")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("N")
plt.ylabel("time(us)")
plt.legend()
plt.show()
