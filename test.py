# 读取csv文件，检查数据并可视化分级分布
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',200)
pd.set_option('display.width', 1000)

df_movies = pd.read_csv('./data/ml-latest-small/movies.csv')

#########  开始使用模型进行推荐  ###################

# 加载模型
from gensim.models import Word2Vec
item2vec_model = Word2Vec.load('item2vec_model_20200416')

def recommender(positive_list=None, negative_list=None, topn=20):
    recommend_movie_ls = []               # most_similar or most_similar_cosmul
    for movieId, prob in item2vec_model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls

query_positive_list = ['1','2'] # 用户喜欢的电影id列表
query_negative_list = None #用户不喜欢的电影id列表
res = recommender(positive_list=query_positive_list, negative_list=query_negative_list, topn=5)
print("Recommendation Result based on:", "\nliked movie:", query_positive_list, "\nunliked movie:", query_negative_list)
print(df_movies[df_movies['movieId'].isin(res)])