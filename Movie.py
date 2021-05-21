from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv('credits.csv')
df2 = pd.read_csv('movies.csv')
df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on='id')
C = df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.9)
# movies having vote count greater than 90 % from the list will be taken
lists_movies = df2.copy().loc[df2['vote_count'] >= m]


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula(m=1838, c=6.09)
    return (v/(v+m) * R) + (m/(m+v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
lists_movies['score'] = lists_movies.apply(weighted_rating, axis=1)

# Sort movies based on score calculated above
lists_movies = lists_movies.sort_values('score', ascending=False)

# Print the top 10 movies
lists_movies[['title_x', 'vote_count', 'vote_average', 'score']].head(10)


pop = df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12, 4))

plt.barh(pop['title_x'].head(6), pop['popularity'].head(
    6), align='center', color='m')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
lists_movies[['title_x']].loc[lists_movies['score'] > 7].head(10)


pop = df2.sort_values('budget', ascending=False)

plt.figure(figsize=(12, 4))

plt.barh(pop['title_x'].head(10), pop['budget'].head(
    10), align='center', color='lightgreen')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("High Budget Movies")
pop = df2.sort_values('revenue', ascending=False)

plt.figure(figsize=(12, 4))

plt.barh(pop['title_x'].head(6), pop['revenue'].head(
    6), align='center', color='lightblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Revenue on Movies")

lists_movies.drop(['title_y'], axis=1, inplace=True)


def get_recommendations(title, cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df2['title_x'].iloc[movie_indices]


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names

    return []


df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

df2[['title_x', 'cast', 'director', 'keywords', 'genres']].head(5)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


df2['soup'] = df2.apply(create_soup, axis=1)


# Import CountVectorizer and create the count matrix

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

pickle.dump(cosine_sim2, open('model1.pkl', 'wb'))
model1 = pickle.load(open('model1.pkl', 'rb'))


# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title_x'])
print(get_recommendations("Diamonds Are Forever", cosine_sim2))
