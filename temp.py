import matplotlib.pyplot as plt
import numpy as np
import torch

# 用于加载PyTorch .pt文件并计算平均值的函数
def load_and_average_pt_file(file_path):
    try:
        # 使用torch.load加载PyTorch张量
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        # 转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # 如果数据形状是(500, 50)，对50个节点求平均
        if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] > 1:
            # 沿着第二个维度（节点维度）计算平均值
            data = np.nanmean(data, axis=1)
        
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return np.array([])  # 返回空数组以防错误

# 加载并平均损失结果
var0_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145506/loss.pt')
var01_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145924/loss.pt')
var02_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-161343/loss.pt')
var03_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-161400/loss.pt')
var04_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-095457/loss.pt')
var05_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-100512/loss.pt')
var1_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132303/loss.pt')
var5_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132323/loss.pt')

# 加载并平均准确率结果
var0_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145506/accuracy.pt')
var01_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145924/accuracy.pt')
var02_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-161343/accuracy.pt')
var03_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-161400/accuracy.pt')
var04_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-095457/accuracy.pt')
var05_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-100512/accuracy.pt')
var1_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132303/accuracy.pt')
var5_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132323/accuracy.pt')

# 打印平均后的数据形状以确认
print(f"平均后 var0_loss_result 形状: {var0_loss_result.shape if hasattr(var0_loss_result, 'shape') else len(var0_loss_result)}")
print(f"平均后 var0_acc_result 形状: {var0_acc_result.shape if hasattr(var0_acc_result, 'shape') else len(var0_acc_result)}")
# 创建x轴（epoch数）
epochs = np.arange(1, 501)  # 假设有500个epoch

# 创建包含两个垂直排列子图的图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# 绘制平均损失曲线
ax1.plot(epochs[:len(var0_loss_result)], var0_loss_result, label='var0')
ax1.plot(epochs[:len(var01_loss_result)], var01_loss_result, label='var01')
ax1.plot(epochs[:len(var02_loss_result)], var02_loss_result, label='var02')  
ax1.plot(epochs[:len(var03_loss_result)], var03_loss_result, label='var03')
ax1.plot(epochs[:len(var04_loss_result)], var04_loss_result, label='var04')
ax1.plot(epochs[:len(var05_loss_result)], var05_loss_result, label='var05')
ax1.plot(epochs[:len(var1_loss_result)], var1_loss_result, label='var1')
ax1.plot(epochs[:len(var5_loss_result)], var5_loss_result, label='var5')  
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Average Loss')
ax1.set_title('Average Loss vs Epochs (Averaged across 50 nodes)')
ax1.legend()
ax1.grid(True)

# 绘制平均准确率曲线
ax2.plot(epochs[:len(var0_acc_result)], var0_acc_result, label='var0')
ax2.plot(epochs[:len(var01_acc_result)], var01_acc_result, label='var01')
ax2.plot(epochs[:len(var02_acc_result)], var02_acc_result, label='var02')  
ax2.plot(epochs[:len(var03_acc_result)], var03_acc_result, label='var03')
ax2.plot(epochs[:len(var04_acc_result)], var04_acc_result, label='var04')
ax2.plot(epochs[:len(var05_acc_result)], var05_acc_result, label='var05')
ax2.plot(epochs[:len(var1_acc_result)], var1_acc_result, label='var1')
ax2.plot(epochs[:len(var5_acc_result)], var5_acc_result, label='var5')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Average Accuracy')
ax2.set_title('Average Accuracy vs Epochs (Averaged across 50 nodes)')
ax2.legend()
ax2.grid(True)

# 调整布局并保存
plt.tight_layout(pad=3.0)  # 增加内边距以解决警告
plt.savefig('node_averaged_loss_and_accuracy.png', dpi=300)
plt.show()