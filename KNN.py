import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader

# 相对路径
path = "mnist课程数据集/mnist to matlab/mat格式的MNIST数据/train_images.mat"
# 数据集
train_datasets = test_datasets = datasets.MNIST(root=path, train=True, download=True)
# 加载数据
batch = 100
train_loader = DataLoader(train_datasets, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batch, shuffle=True)
# 对训练数据处理
x_train = train_loader.dataset.data.numpy()
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_train = x_train.astype(float)
y_train = train_loader.dataset.targets.numpy()
# 对测试数据处理
test_num = 1000   #测试数量
x_test = test_loader.dataset.data[-1 * test_num - 1:-1].numpy()
x_test = np.reshape(x_test, (x_test.shape[0], -1))
x_test = x_test.astype(float)
y_test = test_loader.dataset.targets[-1 * test_num - 1:-1].numpy()
# 数组
k3 = []
k5 = []
k7 = []


def sort(x_test, k, x_train, y_train):
    list_label = []
    num_test = x_test.shape[0]
    for i in range(num_test):
        dis = np.sqrt(np.sum((x_train - np.tile(x_test[i], (x_train.shape[0], 1))) ** 2, axis=1))
        index = np.argsort(dis)[:k]
        class_count = {}
        for j in index:
            class_count[y_train[j]] = class_count.get(y_train[j], 0) + 1
        sorted_class_count = sorted(class_count.items(), key=lambda elem: elem[1], reverse=True)
        list_label.append(sorted_class_count[0][0])
    return np.array(list_label)


# 准确率
with open('output.txt', 'a') as f:
    f.write("Accuracy statistics\n")
for train_num in range(1000, 11000, 1000):
    with open('output.txt', 'a') as f:
        f.write(" train_number: {}\n".format(train_num))
    # 记录k = 3，5，7
    for k in [3, 5, 7]:
        x_trainDatasets = x_train[:train_num]
        y_trainDatasets = y_train[:train_num]
        y_sort = sort(x_test, k, x_trainDatasets, y_trainDatasets)
        num_correct = np.sum(y_sort == y_test)
        accuracy = float(num_correct) / test_num
        if k == 3:
            k3.append(accuracy)
        elif k == 5:
            k5.append(accuracy)
        elif k == 7:
            k7.append(accuracy)
        with open('output.txt', 'a') as f:
            f.write(' k = %d , %d / %d   --> accuracy: %f\n' % (k, num_correct, test_num, accuracy))
