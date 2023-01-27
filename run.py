import json
import plotly
import pandas as pd
import pickle
import joblib
import os 
from flask import send_from_directory   

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
import re
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/Disaster.db')
df = pd.read_sql_table('Data', engine)

# load model
model = joblib.load('models/Model.pkl')

# index webpage displays cool visuals and receives user input text for model

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Percentage of messages falls in each category
    categories = df[df.columns[4:]]
    pct_message = 100*categories.sum().sort_values()/categories.shape[0]
    categories_name = list(pct_message.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': '<b>Distribution of Message Genres</b>',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Category visualization
        {
            'data':[
                Bar(
                    x=categories_name,
                    y=pct_message
                )
            ],
            'layout':{
                'title': '<b>Percentage of Messages in each category</b> <br> (Each message may fall in several categories)',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 25
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():

    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()