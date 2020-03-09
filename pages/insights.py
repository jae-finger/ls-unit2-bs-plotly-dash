# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Imports from this application
from app import app

# 1 column layout
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Insights

            The model featured here uses XGBoost to predict whether an individual would be likely to seek help for anxiety or depression during the PhD program.

            The XGBoost model reported a ~69% accuracy from our test set (20% of the original survey sample). An overview of the results can be seen below:
            
            From the model, the top 3 features based on permutation importance are:
                1. Do you feel discriminated or harassed during graduate school?
                2. Do you feel that you were bullied?
                3. Which gender are you?
            
            As an example, we can see which questions force a student's prediction towards yes (red) or towards no (blue)

            """   
        ),
    ],
    md=7,
)

column2 = dbc.Col(
    [
    html.Br(),
    html.Div(html.Img(src='assets/conf_matrix.PNG', className='img-fluid')),
    html.Br(),
    html.Div(html.Img(src='assets/perm_im.PNG', className='img-fluid')),
    html.Br(),
    html.Div(html.Img(src='assets/shap.png', className='img-fluid')),
    ]
)

layout = dbc.Row([column1, column2])