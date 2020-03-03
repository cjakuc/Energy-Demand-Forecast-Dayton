# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, zipfile, io

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Your Value Proposition

            Emphasize how the app will benefit users. Don't emphasize the underlying technology.

            ✅ RUN is a running app that adapts to your fitness levels and designs personalized workouts to help you improve your running.

            ❌ RUN is the only intelligent running app that uses sophisticated deep neural net machine learning to make your run smarter because we believe in ML driven workouts.

            """
        ),
        dcc.Link(dbc.Button('Your Call To Action', color='primary'), href='/predictions')
    ],
    md=4,
)

# Import the zipped file and unzip it
r = requests.get('https://github.com/cjakuc/DS-Unit2_Build-Week/blob/master/Data/MeanHourly.zip?raw=true')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()
fall_energy = pd.read_table(z.open('fall_energy.csv'),delimiter=',')
spring_energy = pd.read_table(z.open('spring_energy.csv'),delimiter=',')
summer_energy = pd.read_table(z.open('summer_energy.csv'),delimiter=',')
winter_energy = pd.read_table(z.open('winter_energy.csv'),delimiter=',')
fall_temp = pd.read_table(z.open('fall_temp.csv'),delimiter=',')
spring_temp = pd.read_table(z.open('spring_temp.csv'),delimiter=',')
summer_temp = pd.read_table(z.open('summer_temp.csv'),delimiter=',')
winter_temp = pd.read_table(z.open('winter_temp.csv'),delimiter=',')

# Create the figure w/ two side by side subplots
fig = make_subplots(rows=1, cols=2, shared_xaxes=False,
                    subplot_titles=('Mean Hourly MW','Mean Hourly Air Temperature'),
                    x_title='Timestamp')
# Create each individual line for each season of the energy demand
fig.add_trace(go.Scatter(x=summer_energy['timestamp'],
                         y=summer_energy['MW'],
                         line=dict(color='#2CA02D'),
                         name='Summer'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=spring_energy['timestamp'],
                         y=spring_energy['MW'],
                         line=dict(color='#FC7F0F'),
                         name='Spring'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=winter_energy['timestamp'],
                         y=winter_energy['MW'],
                         line=dict(color='#2077B4'),
                         name='Winter'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=fall_energy['timestamp'],
                         y=fall_energy['MW'],
                         line=dict(color='#D72829'),
                         name='Fall'),
              row=1,col=1)
# Create each individual line for each season of the air temperature
fig.add_trace(go.Scatter(x=summer_temp['timestamp'],
                         y=summer_temp['AirTemp'],
                         line=dict(color='#2CA02D'),
                         showlegend=False),
              row=1,col=2)
fig.add_trace(go.Scatter(x=spring_temp['timestamp'],
                         y=spring_temp['AirTemp'],
                         line=dict(color='#FC7F0F'),
                         showlegend=False),
              row=1,col=2)
fig.add_trace(go.Scatter(x=winter_temp['timestamp'],
                         y=winter_temp['AirTemp'],
                         line=dict(color='#2077B4'),
                         showlegend=False),
              row=1,col=2)
fig.add_trace(go.Scatter(x=fall_temp['timestamp'],
                         y=fall_temp['AirTemp'],
                         line=dict(color='#D72829'),
                         showlegend=False),
              row=1,col=2)
# Add a title
fig.update_layout(
    title='Mean Hourly MW vs Mean Hourly Air Temperature at Dayton International Airport (2005-15)'
)
# Update x axis properties
fig.update_xaxes(tickvals=['00:00:00','02:00:00','04:00:00','06:00:00',
                           '08:00:00','10:00:00','12:00:00','14:00:00',
                           '16:00:00','18:00:00','20:00:00','22:00:00'],
                 ticktext=['00:00','02:00','04:00','06:00','08:00','10:00',
                           '12:00','14:00','16:00','18:00','20:00','22:00'],
                 tickangle=45)
# Update y axis properties
fig.update_yaxes(title_text='MW',
                 row=1,col=1)
fig.update_yaxes(title_text='Temperature (F)',
                 row=1,col=2)

column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

layout = dbc.Row([column1, column2])