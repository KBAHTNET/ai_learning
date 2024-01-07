import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('Boston House Prices.csv', delimiter=",")

y_pd = df['MEDV']
X_pd = df.drop(['MEDV'], axis=1)

scaler = StandardScaler()
X_sc = pd.DataFrame(scaler.fit_transform(X_pd), columns=X_pd.columns)
X_np = X_sc.to_numpy()
y_np = y_pd.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

y_train.shape = (y_train.shape[0], 1)
y_test.shape = (y_test.shape[0], 1)

# Определение модели
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # пропускаем данные через слои нс
    # вызывается автоматически в model(X_train)
    def forward(self, x):
        out = self.linear(x)
        return out

# Преобразование данных в тензоры PyTorch
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Определение гиперпараметров
input_size = X_train.shape[1]  # Размерность входных признаков
output_size = 1          # Размерность выхода (одна целевая переменная)
learning_rate = 0.01     # Скорость обучения
num_epochs = 120        # Количество эпох

# Создание экземпляра модели
model = LinearRegression(input_size, output_size)

# Определение функции потерь и оптимизатора
# определяет функцию потерь (MSELoss)
criterion = nn.MSELoss()
# создает оптимизатор (SGD), который будет обновлять параметры модели
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Цикл обучения
for epoch in range(num_epochs):
    
    # Forward pass (прямое распространение)
    
    # получает предсказанные значения
    outputs = model(X_train)
    # вычисляет значение функции потерь между outputs и y_train
    loss = criterion(outputs, y_train)

    # Backward pass и оптимизация
    
    # обнуляет градиенты параметров модели перед backward.
    optimizer.zero_grad()
    # вычисляет градиенты функции потерь по параметрам модели.
    loss.backward()
    # обновляет параметры модели, используя градиенты
    optimizer.step()

    # Вывод промежуточной информации
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Вывод результатов
print('Параметры модели:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.data.numpy()}')


with torch.no_grad():
    predicted = model(X_test).detach().numpy()

def calculate_mse(predicted, actual):

    assert len(predicted) == len(actual), "Длины массивов должны быть одинаковыми."

    # Вычисление суммы квадратов разностей
    squared_errors = np.square(predicted - actual)
    sum_squared_errors = np.sum(squared_errors)

    # Вычисление MSE
    mse = sum_squared_errors / len(predicted)
    return mse

mse = calculate_mse(predicted, y_test.numpy())
print("TEST MSE:", mse)