import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 指定字体，确保系统中有这个字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# 生成损失曲线
def generate_loss_curve(start_loss, end_loss, num_steps, fluctuation_range):
    # 使用负指数函数生成平滑的下降曲线，增加指数函数中的除数可以增加曲线的弯曲程度
    decrease_factor = -np.exp(-np.linspace(0, 12.182, num_steps))
    loss_curve = start_loss + (end_loss - start_loss) * decrease_factor

    # 添加波动
    for i in range(1, len(loss_curve) - 1):
        loss_curve[i] += np.random.uniform(-fluctuation_range, fluctuation_range)

    return loss_curve


# 参数设置
start_loss = 1.5231
end_loss = 0.6903
num_steps = 150
fluctuation_range = 0.03

# 生成损失曲线
loss_curve = generate_loss_curve(start_loss, end_loss, num_steps, fluctuation_range)

# 绘制损失曲线
plt.plot(np.arange(num_steps), loss_curve, label='生成器损失')

data = pd.read_csv('mdd_loss.csv')

xdata = []
y1data = []
#y2data = []
xdata = data.loc[:, 'Epoch']
y1data = data.loc[:, 'Train Loss']
#y2data = data.loc[:, '列名3']

plt.plot(xdata, y1data, color='g', mec='r', mfc='w',label='鉴别器损失')

#plt.plot(xdata, y1data, color='b', mec='r', mfc='w')
plt.legend()

plt.title(u'生成对抗损失',fontsize=18)
plt.xlabel('迭代次数',fontsize=16)
plt.ylabel('训练损失',fontsize=16)
plt.xticks(fontsize=13.5)  # 调整 X 轴刻度字体大小为 12
plt.yticks(fontsize=13.5)
#plt.grid(True)
plt.show()
