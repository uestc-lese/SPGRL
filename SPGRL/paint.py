# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

x = np.array([0.5,1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5])
GClip = np.array([0.5910,0.6820,0.7210,0.7110,0.7130,0.7190,0.7310,0.7220,0.7350,0.7380,0.7780,0.8010,0.8110,0.8100,0.8200,0.8310,0.8130,0.8130,0.8240,0.8180,0.8188])
SCRL = np.array([0.3720,0.4360,0.4650,0.5140,0.5290,0.6170,0.6372,0.6460,0.6540,0.6640,0.7300,0.7320,0.7260,0.7190,0.7090,0.7140,0.7160,0.7160,0.7300,0.7300,0.7210])
GFNN = np.array([0.5390,0.5290,0.5330,0.5240,0.5320,0.5260,0.5140,0.5220,0.5210,0.5260,0.5190,0.5370,0.5270,0.5300,0.5160,0.5200,0.5220,0.5420,0.5210,0.5030,0.5070])
GCN = np.array([0.5230,0.5300,0.5450,0.5500,0.5320,0.5470,0.5430,0.5430,0.5310,0.5240,0.5100,0.4870,0.4730,0.4800,0.4780,0.4840,0.4680,0.4670,0.4690,0.4730,0.4680])
KnnGCN = np.array([0.5720,0.5740,0.5700,0.5650,0.5710,0.5800,0.5860,0.5870,0.5820,0.5880,0.5760,0.5740,0.5690,0.5630,0.5630,0.5660,0.5560,0.5830,0.5630,0.5770,0.5260])

# label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# 线型：-  --   -.  :    ,
# marker：.  ,   o   v    <    *    +    1
plt.figure(figsize=(11, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框


plt.plot(x, GClip, marker='o', color="lightseagreen", label="CMGR", linewidth=1)
plt.plot(x, SCRL, marker='d', color="green", label="SCRL", linewidth=1)
plt.plot(x, GFNN, marker='X', color="red", label="GFNN", linewidth=1)
plt.plot(x, GCN, marker='*', color="blue", label="GCN", linewidth=1)
plt.plot(x, KnnGCN, marker='+', color="plum", label="Knn-GCN", linewidth=1)


group_labels = ['0.01','0.02','0.03','0.04','0.05','0.06','0.07','0.08','0.09','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.5','2.0']  # x轴刻度的标识
plt.xticks(x, group_labels, fontsize=8, fontweight='bold')  # 默认字体大小为10
plt.yticks(fontsize=12, fontweight='bold')
# plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
plt.xlabel("Gaussian noise scale $\lambda$rate", fontsize=13, fontweight='bold')
plt.ylabel("acc (dataset:acm)", fontsize=13, fontweight='bold')
plt.xlim(0.0, 21.2)  # 设置x轴的范围
plt.ylim(0.3, 1)

# plt.legend()          #显示各曲线的图例
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

plt.savefig('./a.png', format='png')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
#plt.show()

