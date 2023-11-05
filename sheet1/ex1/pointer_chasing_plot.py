import matplotlib.pyplot as plt

plt.figure(figsize = (14, 7))
x_0 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456,]

f = open("./result_asta.txt")
lines = f.readlines()
y_list = []
for line in lines:
    list = []
    list = eval(line)
    list = list[1:]
    y_list.append(list)
f.close()

legend = []
j = 0

for i in range(10, 31):
    x = x_0[: i - 2]
    y = y_list[j]
    legend.append('2^%d' %(i))

    plt.plot(x, y, marker = 'o')
    plt.xscale('log')
    plt.yscale('log')
    j = j + 1
    
plt.legend(legend, loc = 'upper right',)
plt.savefig('./pointer_chasing_plot_asta.png')