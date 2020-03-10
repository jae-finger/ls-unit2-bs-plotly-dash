# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import sklearn
from sklearn.model_selection import train_test_split
import dash_daq as daq
import pandas as pd

# Imports from this application
from app import app


#---------------------------------------------------
# Load wrangling function for data

# Create function to clean data
# The data dictionary provided with the article does not match the dataset well. Check readme for explanation as to why I removed some columns. I think the question #'s got mislabeled somehow.

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Predictions

            Fill in the answers to the questions below. 
            Once filled in, the gauge on the right reports the percent chance the student will seek help for anxiety or depression.

            """
        ),
        dcc.Markdown(
            """ ### Gender"""
        ),
        dcc.Dropdown(
            options=[
                {'label': 'Female', 'value': 'Female (including trans female)'},
                {'label': 'Male', 'value': 'Male (including trans male)'}
                ],
    value=''
        ),
        dcc.Slider(
            min=1,
            max=7,
            marks={i: '{}'.format(i) for i in range(10)},
            value=1,
        )
    ],
    md=4,
)

column2 = dbc.Col(
    [
        daq.Gauge(
        id='help-percent',
        max=1,
        min=0,
        value=0.1
        )

    ]
)

layout = dbc.Row([column1, column2])