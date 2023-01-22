# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 15:20:33 2023

@author: pc
"""

#Movie Recommendation System Using IMDB Data
#Item-Based collaborative filtering

import numpy as np
import pandas as pd

column_name = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('users.data', sep='\t', names=column_name)
print(df)
print(len(df))

movie_titles = pd.read_csv("movie_id_titles.csv")
print(movie_titles.head())
print(len(movie_titles))

df = pd.merge(df, movie_titles, on='item_id')
print(df)

movie = df.pivot_table(index='user_id',columns='title',values='rating')
print(movie.head())

#Star Wars user ratings

StarWarsUserRate = movie['Star Wars (1977)'] 
print(StarWarsUserRate)

similar_movie_starwars = movie.corrwith(StarWarsUserRate)

print(similar_movie_starwars)
print(type(similar_movie_starwars))

correlation_starwars = pd.DataFrame(similar_movie_starwars, columns=['Correlation'])
correlation_starwars.dropna(inplace=True)
print(correlation_starwars)

print(correlation_starwars.sort_values('Correlation',ascending=False).head(15))

print(df.drop(['timestamp'], axis=1))

movie_rate = pd.DataFrame(df.groupby('title')['rating'].mean())
print(movie_rate.sort_values('rating', ascending=False).head())

#number of votes

movie_rate['rating_vote_number'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(movie_rate.head())


most_votes = movie_rate.sort_values('rating_vote_number', ascending=False).head()
print(most_votes)

corr_df = correlation_starwars.sort_values('Correlation', ascending=False).head()
print(corr_df)

correlation_starwars = correlation_starwars.join(movie_rate['rating_vote_number'])
print(correlation_starwars.head(10)) 

over100vote = correlation_starwars[correlation_starwars['rating_vote_number']>100].sort_values('Correlation',ascending=False).head(10)                     
print(over100vote)