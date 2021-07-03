import numpy as np
import pandas as pd

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")


ratings = pd.merge(movies, ratings)
#print(ratings.columns)

userRatings = ratings.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
print(userRatings.head())

corrValues = userRatings.corr(method = 'pearson', min_periods = 100)

print(corrValues.head())

myRatings = userRatings.iloc[0].dropna()
print(myRatings)

Sample = pd.Series()
for _ in range(0, len(myRatings.index)):
    data = corrValues[myRatings.index[_]].dropna()
    data = data.map(lambda x : x* (myRatings[_]**2))
    Sample = Sample.append(data)




Sample = Sample.groupby(Sample.index).sum()

Sample.sort_values(inplace=True, ascending=False)
print(Sample.head())

#Final_Sample = Sample.drop(myRatings.index)
#print(Final_Sample.head())
