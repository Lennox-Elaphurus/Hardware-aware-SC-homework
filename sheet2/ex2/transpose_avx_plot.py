import matplotlib.pyplot as plt

plt.figure(figsize = (14, 7))
x = []

f = open("./result_avx.txt")
lines = f.readlines()
y_list = []
for line in lines:
    list = []
    list = eval(line)
    x.append(list[0])
    list = list[1:]
    y_list.append(list)
f.close()

legend = ["expl vec", "intrinsics", "non-temporal"]
for i in range(3):
    y = []
    for line in y_list:
        y.append(line[i])
    plt.plot(x, y, marker = 'o')
    plt.xscale('log')
    plt.yscale('log')

plt.legend(legend, loc = 'upper right',)
plt.savefig('./transpose_avx_plot.png')