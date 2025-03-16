import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件（请根据实际情况调整文件路径以及是否存在表头）
data = pd.read_csv('/data/qiaoq/Project/salad_tz/train_result/lightning_logs/version_5/metrics.csv')

# 如果 CSV 文件没有表头，可以这样指定列名：
# data = pd.read_csv('data.csv', header=None, names=['θ', 'loss', 'b_acc', 'step'])

# 创建图形并绘制 loss 曲线
plt.figure(figsize=(10, 6))
plt.plot(data['step'], data['loss'], marker='o', linestyle='-', label='Loss')
plt.xlabel('batch')
plt.ylabel('Loss')
plt.title('Loss 曲线')
plt.legend()
plt.grid(True)

# 保存图像为 JPG 格式
plt.savefig('/data/qiaoq/Project/salad_tz/train_result/lightning_logs/version_5/loss_curve.jpg', format='jpg')
plt.close()  # 关闭图形