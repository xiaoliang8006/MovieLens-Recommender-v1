#########################################################################
# item2vec算法 参考链接：https://blog.csdn.net/fuzi2012/article/details/91345164
#########################################################################

# 读取csv文件，检查数据并可视化分级分布
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',200)
pd.set_option('display.width', 1000)

df_movies = pd.read_csv('./data/ml-latest-small/movies.csv')
df_ratings = pd.read_csv('./data/ml-latest-small/ratings.csv')

# Randomly display 5 records in the dataframe
# for df in list((df_movies, df_ratings)):
#     rand_idx = np.random.choice(len(df), 5, replace=False)
#     print(df.iloc[rand_idx,:])
#     print("Displaying 5 of the total "+str(len(df))+" data points")

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
df_ratings_train, df_ratings_test= train_test_split(df_ratings,stratify=df_ratings['userId'],random_state = 15688,test_size=0.20)

"""
为了让模型学习item embedding，需要从数据中获取“单词”和“句子”等价物。
在这里，把每个“电影”看做是一个“词”，并且从用户那里获得相似评级的电影都在同一个“句子”中。
具体来说，“句子”是通过以下过程生成的：为每个用户生成2个列表，分别存储用户“喜欢”和“不喜欢”的电影。
第一个列表包含所有的电影评级为4分或以上。第二个列表包含其余的电影。假设有n个用户，则这2n个列表就是训练gensim word2vec模型的输入了。
"""
def rating_splitter(df):
    df['liked'] = np.where(df['rating'] >= 4, 1, 0)
    df['movieId'] = df['movieId'].astype('str')
    gp_user_like = df.groupby(['userId', 'liked'])
    return ([gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups])

pd.options.mode.chained_assignment = None # 这个设置会关闭掉copywarning，也有人提问到关闭这个warning过后，速度更快，有待验证
splitted_movies = rating_splitter(df_ratings_train)
# print(splitted_movies)


##########  利用Gensim训练item2vec的模型  ##########

"""
对于原来的word2vec，窗口大小会影响我们搜索“上下文”以定义给定单词含义的范围。按照定义，窗口的大小是固定的。
但是，在item2vec实现中，电影的“含义”应该由同一列表中的所有邻居捕获。换句话说，我们应该考虑用户“喜欢”的所有电影，以定义这些电影的“含义”。
这也适用于用户“不喜欢”的所有电影。然后需要根据每个电影列表的大小更改窗口大小。为了在不修改gensim模型的底层代码的情况下解决这个问题，
首先指定一个非常大的窗口大小，这个窗口大小远远大于训练样本中任何电影列表的长度。然后，在将训练数据输入模型之前对其进行无序处理，
因为在使用“邻近”定义电影的“含义”时，电影的顺序没有任何意义。
Gensim模型中的窗口参数实际上是随机动态的。我们指定最大窗口大小，而不是实际使用的窗口大小。尽管上面的解决方法并不理想，但它确实实现了可接受的性能。
最好的方法可能是直接修改gensim中的底层代码，但这就超出了我目前的能力范围了，哈哈哈。
"""

# 打乱电影顺序
import random
for movie_list in splitted_movies:
    random.shuffle(movie_list)

# 开始训练模型
from gensim.models import Word2Vec
import datetime
start = datetime.datetime.now()
item2vec_model = Word2Vec(sentences = splitted_movies,
                        iter = 1, # epoch
                        min_count = 5, # a movie has to appear more than 5 times to be keeped
                        size = 300, # size of the hidden layer
                        workers = 20, # specify the number of threads to be used for training
                        sg = 1, #用于设置训练算法。当 sg=0，使用 CBOW 算法来进行训练；当 sg=1，使用 skip-gram 算法来进行训练
                        hs = 0,
                        negative = 5, #负采样
                        window = 9999999)
end = datetime.datetime.now()
print("Total time: " + str(end-start))
item2vec_model.save('item2vec_model_20200416')
# del model_w2v_sg


#########  开始使用模型进行推荐  ###################

def recommender(positive_list=None, negative_list=None, topn=20):
    recommend_movie_ls = []               # most_similar or most_similar_cosmul
    for movieId, prob in item2vec_model.wv.most_similar(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls

query_positive_list = ['1','2'] # 用户喜欢的电影id列表
query_negative_list = None #用户不喜欢的电影id列表
res = recommender(positive_list=query_positive_list, negative_list=query_negative_list, topn=5)
print("Recommendation Result based on:", "\nliked movie:", query_positive_list, "\nunliked movie:", query_negative_list)
print(df_movies[df_movies['movieId'].isin(res)])
