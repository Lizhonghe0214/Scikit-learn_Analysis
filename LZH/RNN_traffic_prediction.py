"""
采用pytorch框架，并以sklearn库为辅助，利用循环神经网络进行出租车费用流量预测
"""

import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# 预备数据清洗函数
def clean_data(data):
    data = data[data["total_amount"] > 0]
    data = data[data["total_amount"] < 100]
    return data


# 导入数据
df = pd.read_csv('datasets/green_tripdata_2016-12.csv')
# 取前1000000行数据
df = df[:1000000]
# 筛选出需要的数据列
df = df[['lpep_pickup_datetime', 'total_amount']]
# 转换日期列为datetime格式
df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
# 处理缺失值，删除缺失出发时间、总金额等的行
df = df.dropna(subset=['lpep_pickup_datetime', 'total_amount'])
# 清洗数据
df = clean_data(df)
# 数据变形
df.set_index('lpep_pickup_datetime', inplace=True)
# 车费数据按每60min聚合
df_resampled = df.resample('60T').agg({'total_amount': 'sum'})
# 查看处理后数据
print(df_resampled[:10])
print(len(df_resampled))
# 分离输入和输出
X = df_resampled.index.values.astype('int64') // 10 ** 11  # 转换为 Unix 时间戳（秒数）
X = X.reshape(-1, 1)
y = df_resampled.values
print(X[:5])
print(y[:5])

# 可视化每60分钟的出租车费用流量数据
plt.figure(figsize=(12, 6))
plt.plot(X, y, label='total amount')
plt.xlabel('Date Time')
plt.ylabel('total amount')
plt.title('Total Amount Every 60 Minutes')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 数据标准化
y = y.astype('float32')
min_max_scaler = MinMaxScaler()
data = min_max_scaler.fit_transform(y)
print(data[:5])


# 创建数据集
def create_dataset(dataset, look_back=10):
    data_X, data_y = [], []
    for i in range(len(dataset) - look_back):
        temp = dataset[i:(i + look_back)]
        data_X.append(temp)
        data_y.append(dataset[i + look_back])
    return np.array(data_X), np.array(data_y)


# 输入序列长度
sequence_length = 24
data_X, data_y = create_dataset(data, sequence_length)
print(data_X[:3])
print(data_y[:3])

# 划分训练集和测试集（7：3）
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_y = data_y[:train_size]
test_X = data_X[train_size:]
test_y = data_y[train_size:]

# 数据变形（Tensor）
train_X = torch.from_numpy(train_X).float().view(-1, sequence_length, 1)
train_y = torch.from_numpy(train_y).float().view(-1, 1)
test_X = torch.from_numpy(test_X).float().view(-1, sequence_length, 1)


# 定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1
        hidden_size = 64
        num_layers = 2
        output_size = 1
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


# 定义LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1
        hidden_size = 64
        num_layers = 2
        output_size = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


# 定义GRU模型
class SimpleGRU(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 1
        hidden_size = 64
        num_layers = 2
        output_size = 1
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out


# 实例化三种循环神经网络模型
model_lstm = SimpleLSTM()
model_rnn = SimpleRNN()
model_gru = SimpleGRU()
# 实例化对应的优化器
optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.01)
optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=0.01)
optimizer_gru = torch.optim.Adam(model_gru.parameters(), lr=0.01)
# 实例化损失函数
criterion = nn.MSELoss()
# 训练轮数
num_epochs = 500


# 训练函数
def model_train(model, optimizer, criterion, X, y, num_epochs):
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}],Loss:{loss.item():.4f}')


# 预测函数
def model_predict(model, X, start, end):
    model.eval()
    predicted = []
    with torch.no_grad():
        test_input = X[:1]
        for _ in range(end - start):
            output = model(test_input)
            predicted.append(output.item())
            test_input = torch.cat((test_input[:, 1:, :], output.view(1, 1, 1)), dim=1)
    return predicted


# lstm
print('----------LSTM----------')
model_train(model_lstm, optimizer_lstm, criterion, train_X, train_y, num_epochs)
predicted_lstm = model_predict(model_lstm, test_X, len(train_X), len(data_X))
mse_lstm = mean_squared_error(test_y, predicted_lstm)
print(f'LSTM MSE:{mse_lstm}')
print(len(test_y), len(predicted_lstm))
print(len(data), len(train_X), len(predicted_lstm))
# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(np.arange(len(train_X) + 24, len(train_X) + len(predicted_lstm) + 24), predicted_lstm, label='Predicted')
plt.legend()
plt.show()

# rnn
print('----------RNN----------')
model_train(model_rnn, optimizer_rnn, criterion, train_X, train_y, num_epochs)
predicted_rnn = model_predict(model_rnn, test_X, len(train_X), len(data_X))
mse_rnn = mean_squared_error(test_y, predicted_rnn)
print(f'RNN MSE:{mse_rnn}')
# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(np.arange(len(train_X) + 24, len(train_X) + len(predicted_rnn) + 24), predicted_rnn, label='Predicted')
plt.legend()
plt.show()

# gru
print('----------GRU----------')
model_train(model_gru, optimizer_gru, criterion, train_X, train_y, num_epochs)
predicted_gru = model_predict(model_gru, test_X, len(train_X), len(data_X))
mse_gru = mean_squared_error(test_y, predicted_gru)
print(f'GRU MSE:{mse_gru}')
# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data, label='Actual')
plt.plot(np.arange(len(train_X) + 24, len(train_X) + len(predicted_gru) + 24), predicted_gru, label='Predicted')
plt.legend()
plt.show()
