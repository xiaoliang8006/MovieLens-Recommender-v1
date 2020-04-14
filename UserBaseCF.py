########################################################
# 参考链接：https://cloud.tencent.com/developer/article/1092296
########################################################
# 1.基于用户的协同过滤
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 2.数据初探
ratings = pd.read_csv('./ml-latest-small/ratings.csv',index_col=None)
# print(ratings.head(5))
movies = pd.read_csv('./ml-latest-small/movies.csv',index_col=None)
# print(movies.head(5))

"""
# 我们可以根据movieId来合并两个数据集
data = pd.merge(ratings,movies,on='movieId')

# 合并数据集之后，我们可以看一下每部电影的评分数量，并按照降序进行排序：
rating_count_by_movie = data.groupby(['movieId','title'],as_index=False)['rating'].count()
rating_count_by_movie.columns=['movieId','title','rating_count']
rating_count_by_movie.sort_values(by=['rating_count'],ascending=False,inplace=True)
print(rating_count_by_movie[:10])

# 得到打分的平均值及方差
rating_stddev = data.groupby(['movieId','title']).agg({'rating':['mean','std']})
print(rating_stddev.head(10))
"""

# 3.数据预处理
moviesPath = './ml-latest-small/movies.csv'
ratingsPath = './ml-latest-small/ratings.csv'
moviesDF = pd.read_csv(moviesPath,index_col=None)
ratingsDF = pd.read_csv(ratingsPath,index_col=None)

# 这里我们按照4:1的比例将数据集进行拆分
trainRatingsDF,testRatingsDF = train_test_split(ratingsDF,test_size=0.2)

# 接下来，我们得到用户-电影的评分同现矩阵，使用pandas的数据透视功能，同时，我们得到电影id和用户id与其对应索引的映射关系：
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId','movieId','rating']],
                                     columns=['movieId'],index=['userId'],values='rating',fill_value=0)
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
ratingValues = trainRatingsPivotDF.values.tolist() # 将每行(每个用户)信息存入列表
# print(trainRatingsPivotDF)
# print(moviesMap)
# print(usersMap)


# 4.用户相似度计算
# 这里我们使用余弦相似度来计算用户之间的相似度关系
from sklearn.metrics import pairwise_distances
userSimMatrix = 1-pairwise_distances(ratingValues, metric="cosine") #pairwise_distances计算非常快
# print(userSimMatrix[0])


# 5.电影推荐
# 接下来，我们要找到与每个用户最相近的K个用户，用这K个用户的喜好来对目标用户进行物品推荐，这里K=10，下面的代码用来计算与每个用户最相近的10个用户：
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userSimMatrix[i][i] = 0 #先将自己去除
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])),key = lambda x:x[1],reverse=True)[:10]


# 得到了每个用户对应的10个兴趣最相近的用户之后，我们对其他相似用户加权计算用户对每个没有观看过的电影的兴趣分
# 这里，如果用户已经对电影打过分，那么兴趣值就是0
userRecommendValues = np.zeros((len(ratingValues),len(ratingValues[0])),dtype=np.float32)
for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
            if ratingValues[i][j] == 0:
                val = 0
                sim_sum = 0
                for (user,sim) in userMostSimDict[i]:
                    val += (ratingValues[user][j] * sim)
                    sim_sum += sim
                userRecommendValues[i,j] = val/sim_sum

# 接下来，我们为每个用户推荐10部电影：
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])),key = lambda x:x[1],reverse=True)[:10]


###### 以下是为了输出好看。。。##########
# 这里要注意的是，我们一直使用的是索引，我们需要将索引的用户id和电影id转换为真正的用户id和电影id，这里我们前面定义的两个map就派上用场了
userRecommendList = []
for key,value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId,val) in value:
        userRecommendList.append([user,moviesMap[movieId]])

# 最后一步，我们将推荐结果的电影id转换成对应的电影名，并打印结果：
recommendDF = pd.DataFrame(userRecommendList,columns=['userId','movieId'])
recommendDF = pd.merge(recommendDF,moviesDF[['movieId','title']],on='movieId',how='inner')
print(recommendDF.tail(10))
