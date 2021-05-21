from flask import Flask, request, render_template
import pickle
import pandas as pd


app = Flask(__name__, template_folder='templates', static_folder='static')
model1 = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('recommendation.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    output = []
    df1 = pd.read_csv("credits.csv")
    df2 = pd.read_csv("movies.csv")
    df1.columns = ['id', 'title', 'cast', 'crew']
    df2 = df2.merge(df1, on='id')
    req = request.form
    final1 = req.get("Temperature")
    final = final1.title()

    def get_recommendations(title, cosine_sim):
        indices = pd.Series(df2.index, index=df2['title_x']).drop_duplicates()
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return df2['title_x'].iloc[movie_indices]
    output.extend(get_recommendations(final, model1))
    output = "  ,  ".join(output)
    return render_template('recommendation.html', pred=output, end='\n')


if __name__ == '__main__':
    app.run(debug=True)
