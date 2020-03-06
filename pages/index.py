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
        
            # Energy Demand Predictions

            With this app you can predict the energy demand for the Dayton, Ohio region for Dayton Power 
            and Light Company. On the right are two figures that visualize how energy demand in this area 
            related to time, temperature, and season between 2005 and 2015. They are also interactive and 
            you can see the values for individual observations via hovering with your cursor. You can zoom 
            on both figures by selecting a section with your cursor. Double-clicking resets the plot after zooming. \n

            Try running some predictions using different models and features by clicking the predictions button below!

            """
        ),
    ],
    md=4,
)
column3 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            

            """
        ),
        dcc.Link(dbc.Button('Predictions', color='primary'), href='/predictions')
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
                         name='Summer',
                         hovertext='(Time, MW)'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=spring_energy['timestamp'],
                         y=spring_energy['MW'],
                         line=dict(color='#FC7F0F'),
                         name='Spring',
                         hovertext='(Time, MW)'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=winter_energy['timestamp'],
                         y=winter_energy['MW'],
                         line=dict(color='#2077B4'),
                         name='Winter',
                         hovertext='(Time, MW)'),
              row=1,col=1)
fig.add_trace(go.Scatter(x=fall_energy['timestamp'],
                         y=fall_energy['MW'],
                         line=dict(color='#D72829'),
                         name='Fall',
                         hovertext='(Time, MW)'),
              row=1,col=1)
# Create each individual line for each season of the air temperature
fig.add_trace(go.Scatter(x=summer_temp['timestamp'],
                         y=summer_temp['AirTemp'],
                         line=dict(color='#2CA02D'),
                         showlegend=False,
                         name='Summer',
                         hovertext='(Time, Temperature)'),
              row=1,col=2)
fig.add_trace(go.Scatter(x=spring_temp['timestamp'],
                         y=spring_temp['AirTemp'],
                         line=dict(color='#FC7F0F'),
                         showlegend=False,
                         name='Spring',
                         hovertext='(Time, Temperature)'),
              row=1,col=2)
fig.add_trace(go.Scatter(x=winter_temp['timestamp'],
                         y=winter_temp['AirTemp'],
                         line=dict(color='#2077B4'),
                         showlegend=False,
                         name='Winter',
                         hovertext='(Time, Temperature)'),
              row=1,col=2)
fig.add_trace(go.Scatter(x=fall_temp['timestamp'],
                         y=fall_temp['AirTemp'],
                         line=dict(color='#D72829'),
                         showlegend=False,
                         name='Fall',
                         hovertext='(Time, Temperature)'),
              row=1,col=2)
# Add a title
title = {'xref':'paper', 'yref':'paper', 'x':0.0,
                              'xanchor':'left', 'yanchor':'bottom',
                              'text':'Mean Hourly MW vs Mean Hourly Air Temperature (2005-2015)',
                              'font':dict(family='Arial',
                                        size=20)}
fig.update_layout(title=title,template='plotly_dark')
# Update x axis properties
fig.update_xaxes(tickvals=['00:00:00','02:00:00','04:00:00','06:00:00',
                           '08:00:00','10:00:00','12:00:00','14:00:00',
                           '16:00:00','18:00:00','20:00:00','22:00:00'],
                 ticktext=['00:00','02:00','04:00','06:00','08:00','10:00',
                           '12:00','14:00','16:00','18:00','20:00','22:00'],
                 tickangle=45)
# Update y axis properties
fig.update_yaxes(title_text='MW', patch=dict(title=dict(standoff=0)),
                 row=1,col=1)
fig.update_yaxes(title_text='Temperature (F)', patch=dict(title=dict(standoff=0)),
                 row=1,col=2)
# Update line widths
fig.update_traces(line=dict(width=3))
# Add footer
annotations = []
# Add footer
fig.add_annotation(dict(xref='paper', yref='paper', x=-0.1, y=-0.25,
                              xanchor='left', yanchor='bottom',
                              text='Source: data from DPL and NOAA',
                              font=dict(family='Arial',
                                        size=10,
                                        color='gray'),
                              showarrow=False))

# Import the data for the second figure
daily_energy_averages = pd.read_csv('https://github.com/cjakuc/DS-Unit2_Build-Week/raw/master/Data/daily_energy.csv',index_col='date')

# Create the second figure
# Create an array of the last day of each season
season_end = [59,151,243,334]
season_x_labels=['January 1st',
                 'February 19th',
                 'April 10th',
                 'May 30th',
                 'July 19th',
                 'September 7th',
                 'October 27th',
                 'December 16th']
season_x_ticks = [1,50,100,150,200,250,300,350]

# Make the plot
fig1 = px.line(daily_energy_averages,
              x=daily_energy_averages.index.get_level_values('date'),
              y='MW',
              labels={'x':'Day'},
              template = 'plotly_dark')
# Change the color of the line
fig1.update_traces(line=dict(color='gold',width=2))
# Add vertical lines at season beginning/endings
season_list = [] # Create a list of dict objects to create each individual line
for season in season_end:
  season_list.append(dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= season, x1= season,
      line=dict(color='green',dash='dash'),
      name='Season Change'
      ))
season_tuple = tuple(season_list) # shapes= expects a tuple so convert list to tuple
# Add dotted lines
fig1.update_layout(shapes=season_tuple)
# Add annotations for title and holidays
annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Mean Total MW Consumed per Day (2005-2015)',
                              font=dict(family='Arial',
                                        size=20),
                              showarrow=False))
# Add subtitle
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.00,
                              xanchor='left', yanchor='bottom',
                              text='Dashed lines show change of seasons',
                              font=dict(family='Arial',
                                        size=15,
                                        color='gray'),
                              showarrow=False))
# Add footer
annotations.append(dict(xref='paper', yref='paper', x=-0.1, y=-0.25,
                              xanchor='left', yanchor='bottom',
                              text='Source: data from DPL',
                              font=dict(family='Arial',
                                        size=10,
                                        color='gray'),
                              showarrow=False))
  # Holidays
annotations.append(dict(
    x=185, y=1900, text="4th of July", font=dict(size=12),
    xref="x",yref="y", showarrow=True, arrowhead=7, ax=10, ay=40,))
annotations.append(dict(
    x=359, y=1770, text="Christmas", font=dict(size=12),
    xref="x",yref="y", showarrow=True, arrowhead=7, ax=-77, ay=0))
annotations.append(dict(
    x=1, y=1835, text="New Years Day", font=dict(size=12),
    xref="x",yref="y", showarrow=True, arrowhead=7, ax=42, ay=40))
fig1.update_layout(annotations=annotations,showlegend=True)
# Update x axis
fig1.update_xaxes(
                 tickvals=season_x_ticks,
                 ticktext=season_x_labels,
                 tickangle=15)
# Update y axis properties
fig1.update_yaxes(title_text='MW', patch=dict(title=dict(standoff=0)))

column2 = dbc.Col(
    [
        dcc.Graph(figure=fig1),
    ]
)

column4 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)

layout = html.Div(
    [
        dbc.Row([column1, column2]),
        dbc.Row([column3, column4])
    ]
)