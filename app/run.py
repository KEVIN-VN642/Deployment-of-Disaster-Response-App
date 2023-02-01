import json
import plotly
import pandas as pd
import joblib
import os
import sys 
from flask import send_from_directory   

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
import re
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
sys.path.append('..')

from models.Utils import tokenize

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/Disaster.db')
df = pd.read_sql_table('Data', engine)

# load model
model = joblib.load('../models/Model.pkl')

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
    genre_pct = round(100*genre_counts/genre_counts.sum(),2)
    
    # Percentage and number of messages falls in each category
    categories = df[df.columns[4:]]
    category_counts = categories.sum().sort_values()
    pct_message = round(100*categories.sum().sort_values()/categories.shape[0],2)
    categories_name = list(pct_message.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            "data":[
                {
                    "type": "pie",
                    "hole": 0.45,
                    "name": "Genre",
                    "pull": 0,
                    "marker": {
                        "colors":[
                            "#85cc81",
                            "#c293bc",
                            "#c9c740"
                        ]
                    },
                    "textinfo": "label+value",
                    "hoverinfo": "all",
                    "labels": genre_names,
                    "values": genre_counts
                }
            ],
            "layout": {
                "title": "<b>Count and Percentage of Messages by Genre</b>"
            }

        },
        # Category visualization - percentage
        {
            'data':[
                {
                    "type": "bar",
                    "x": categories_name,
                    "y": pct_message,
                    "marker": {
                        "color": "#1e96e6"
                    }
                }
            ],
            'layout':{
                'title': '<b>Percentage of Messages in each category</b> <br> (Each message may fall in several categories)',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 30
                },
                "margin": {
                    "b": 150
                },
                "height": 500
            }
        },
        # Category visualization - count
        {
            'data':[
                {
                    "type": "bar",
                    "x": categories_name,
                    "y": category_counts,
                    "marker": {
                        "color": "#1e96e6"
                    }
                }
            ],
            'layout':{
                'title': '<b>Number of Messages in each category</b> <br> (Each message may fall in several categories)',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': 'Category',
                    'tickangle': 30
                },
                "margin": {
                    "b": 150
                },
                "height": 500
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
    clean_query = tokenize(query)

    # use model to predict classification for query
    classification_labels = model.predict([clean_query])[0]
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
