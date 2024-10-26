import csv
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('mdd_loss.csv')

xdata = []
y1data = []
#y2data = []
xdata = data.loc[:, 'Epoch']
y1data = data.loc[:, 'Train Loss']
#y2data = data.loc[:, '列名3']

plt.plot(xdata, y1data, color='b', mec='r', mfc='w')
plt.tick_params(labelsize=12)
#plt.plot(xdata, y2data, color='b', marker='o', mec='r', mfc='w', label=u'列名3')  # color可自定义折线颜色，marker可自定义点形状，label为折线标注
plt.title(u"多尺度变分自编码器", size=15)
plt.legend()
plt.xlabel(u'Epoch', size=15)
plt.ylabel(u'Training Loss', size=15)

plt.show()
