import matplotlib.pyplot as plt

plt.figure(figsize = (14, 7))
x = []
#x = [24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576]

f = open("./result.txt")
lines = f.readlines()
y_list = []
for line in lines:
    list = []
    list = eval(line)
    x.append(list[0])
    list = list[1:]
    y_list.append(list)
f.close()

legend = ["vanilla-consecutive-write", "vanilla-strided-write", "blocked-consecutive-write-M4", "blocked-strided-writed-M4", "blocked-consecutive-write-M8", "blocked-strided-writed-M8", "blocked-consecutive-write-M12", "blocked-strided-writed-M12"]
for i in range(8):
    y = []
    for line in y_list:
        y.append(line[i])
    plt.plot(x, y, marker = 'o')
    plt.xscale('log')
    plt.yscale('log')

plt.legend(legend, loc = 'upper right',)
plt.savefig('./transpose_plot.png')