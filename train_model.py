import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. 加载训练数据
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 2. 训练线性模型
model = LinearRegression()
model.fit(X, y)

# 3. 保存模型二进制文件
joblib.dump(model, 'linear_model.pkl')

# 4. 将模型系数保存到文本文件（用于 Release 附件展示）
with open('linear_model.txt', 'w') as f:
     f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')
    
print("模型训练完成，已生成 linear_model.txt")