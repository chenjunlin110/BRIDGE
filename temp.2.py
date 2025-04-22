import matplotlib.pyplot as plt
import numpy as np
import torch
def load_and_average_pt_file(file_path):
    try:
        # 使用torch.load加载PyTorch张量
        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=True)
        # 转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return np.array([])  # 返回空数组以防错误
    

# var05_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-100512/loss.pt')
# var05_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250419-100512/accuracy.pt')

# var1_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132303/loss.pt')
# var1_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250420-132303/accuracy.pt')

var0_loss_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145506/loss.pt')
var0_acc_result = load_and_average_pt_file('/Users/junlinchen/Desktop/code/learning/BRIDGE/results_BRIDGE-T_20250418-145506/accuracy.pt')
labels = ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7', 'node_8', 'node_9'
          'node_10', 'node_11', 'node_12', 'node_13', 'node_14', 'node_15', 'node_16', 'node_17', 'node_18',
          'node_19', 'node_20', 'node_21', 'node_22', 'node_23', 'node_24', 'node_25', 'node_26', 'node_27',
          'node_28', 'node_29', 'node_30', 'node_31', 'node_32', 'node_33', 'node_34', 'node_35', 'node_36',
          'node_37', 'node_38', 'node_39', 'node_40', 'node_41', 'node_42', 'node_43', 'node_44', 'node_45',
          'node_46', 'node_47', 'node_48', 'node_49']
fig, (ax1, ax2) = plt.subplots(2, 1)
for data, label in zip(var0_loss_result.T, labels):
    ax1.plot(data, label=label)
for data, label in zip(var0_acc_result.T, labels):
    ax2.plot(data, label=label)
plt.title('Loss and Accuracy Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid()
plt.show()




