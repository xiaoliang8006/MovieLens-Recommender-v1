# MovieLens数据集

MovieLens数据集是一个关于电影评分的数据集，里面包含了从IMDB, The Movie DataBase上面得到的用户对电影的评分信息，详细请看下面的介绍。

## links.csv:

文件里面的内容是帮助你如何通过网站id在对应网站上找到对应的电影链接的。数据格式如下：

movieId, imdbId, tmdbId

	movieId:表示这部电影在movielens上的id，可以通过链接https://movielens.org/movies/(movieId)来得到。
	imdbId:表示这部电影在imdb上的id，可以通过链接http://www.imdb.com/title/(imdbId)/
	来得到。
	tmdbId:表示这部电影在themoviedb上的id，可以通过链接http://www.imdb.com/title/(tmdbId)/
	来得到。


## movies.csv:

文件里包含了一部电影的id和标题，以及该电影的类别。数据格式如下：

movieId, title, genres

	movieId:每部电影的id
	title:电影的标题
	genres:电影的类别（详细分类见readme.txt）

## ratings.csv:

文件里面的内容包含了每一个用户对于每一部电影的评分。数据格式如下：

userId, movieId, rating, timestamp

	userId: 每个用户的id
	movieId: 每部电影的id
	rating: 用户评分，是5星制，按半颗星的规模递增(0.5 stars - 5 stars)
	timestamp: 自1970年1月1日零点后到用户提交评价的时间的秒数

数据排序的顺序按照userId，movieId排列的。

## tags.csv:
文件里面的内容包含了每一个用户对于每一个电影的分类。数据格式如下：

userId, movieId, tag, timestamp

	userId: 每个用户的id
	movieId: 每部电影的id
	tag: 用户对电影的标签化评价
	timestamp: 自1970年1月1日零点后到用户提交评价的时间的秒数

数据排序的顺序按照userId，movieId排列的。

## 下载链接：
官网地址: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

# 模型

使用了UserbaseCF模型、ItemBaseCF模型以及Item2vec模型。我们首先得到用户-电影的评分同现矩阵，然后使用余弦相似度来计算用户之间或商品之间的相似度关系。接下来，我们要找到与每个用户(商品)最相近的K个用户(商品)，用这K个用户(商品)来对目标用户(商品)进行推荐。

Item2vec模型中，我们首先用Word2vec得到电影的embedding,然后再进行相似计算。

**具体过程代码中有非常详细的注释。**
