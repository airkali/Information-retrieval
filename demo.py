import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# 加载数据集
ratings = pd.read_csv('datas/ratings.csv')
movies = pd.read_csv('datas/movies.csv')

# 数据预处理
ratings = ratings.dropna()
movies = movies.dropna()

# 获取用户和项目的数量
num_users = ratings['userId'].nunique()
num_items = ratings['movieId'].nunique()



# 划分数据集
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# 实现矩阵分解
U, sigma, Vt = np.linalg.svd(train_data, full_matrices=False)

# 创建用户和项目的映射
user_mapping = {id:i for i, id in enumerate(ratings['userId'].unique())}
item_mapping = {id:i for i, id in enumerate(ratings['movieId'].unique())}

def sgd_bias(data, k, max_epochs, alpha, lambda_reg):
    # 初始化用户和项目的潜在因子矩阵
    P = np.random.normal(0, .1, (num_users, k))
    Q = np.random.normal(0, .1, (num_items, k))
    
    # 初始化全局偏置、用户偏置和项目偏置
    mu = np.mean(data['rating'])
    bu = np.zeros(num_users)
    bi = np.zeros(num_items)

    # 进行SGD
    for epoch in range(max_epochs):
        for row in data.itertuples():
            u, i, r = user_mapping[row.userId], item_mapping[row.movieId], row.rating
            e = r - (mu + bu[u] + bi[i] + np.dot(P[u, :], Q[i, :]))
            bu[u] += alpha * (e - lambda_reg * bu[u])
            bi[i] += alpha * (e - lambda_reg * bi[i])
            P[u, :] += alpha * (e * Q[i, :] - lambda_reg * P[u, :])
            Q[i, :] += alpha * (e * P[u, :] - lambda_reg * Q[i, :])

    return mu, bu, bi, P, Q

# 调用SGD函数
mu, bu, bi, P, Q = sgd_bias(train_data, k=5, max_epochs=200, alpha=0.02, lambda_reg=0.05)

# 在测试集上评估模型
test_data['pred'] = test_data.apply(lambda row: mu + bu[user_mapping[row.userId]] + bi[item_mapping[row.movieId]] + np.dot(P[user_mapping[row.userId], :], Q[item_mapping[row.movieId], :]), axis=1)
mse = np.mean((test_data['rating'] - test_data['pred']) ** 2)
print('Test MSE:', mse)



# 计算预测误差
test_data['error'] = test_data['rating'] - test_data['pred']

# 保存预测误差的直方图
plt.figure(figsize=(8, 8))
plt.hist(test_data['error'], bins=50, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.savefig('error_histogram.png')

# 绘制前100个评分和预测评分的折线图
plt.figure(figsize=(8, 8))
plt.plot(test_data['rating'].values[:100], label='Original Ratings')
plt.plot(test_data['pred'].values[:100], label='Predicted Ratings')
plt.xlabel('Index')
plt.ylabel('Rating')
plt.title('Original vs. Predicted Ratings for the First 100 Test Samples')
plt.legend()
plt.grid(True)
plt.savefig('ratings_comparison.png')
plt.show()