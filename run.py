# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# from pages.predictions import predict

# Imports from this application
from app import app, server
from pages import predictions
from joblib import load

# Load pages pickles so everything doesn't run dynamically
index = load('assets/index')
# predictions = load('assets/predictions')
process = load('assets/process')
predictions_dict = load('assets/predictions_dict')


# Navbar docs: https://dash-bootstrap-components.opensource.faculty.ai/l/components/navbar
navbar = dbc.NavbarSimple(
    brand='Forecasting Energy Demand in Dayton, Ohio',
    brand_href='/', 
    children=[
        dbc.NavItem(dcc.Link('Predictions and Insights', href='/predictions', className='nav-link')), 
        # dbc.NavItem(dcc.Link('Insights', href='/insights', className='nav-link')), 
        dbc.NavItem(dcc.Link('Process', href='/process', className='nav-link')), 
    ],
    sticky='top',
    color='light', 
    light=True, 
    dark=False
)

# Footer docs:
# dbc.Container, dbc.Row, dbc.Col: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
# html.P: https://dash.plot.ly/dash-html-components
# fa (font awesome) : https://fontawesome.com/icons/github-square?style=brands
# mr (margin right) : https://getbootstrap.com/docs/4.3/utilities/spacing/
# className='lead' : https://getbootstrap.com/docs/4.3/content/typography/#lead
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span('Chris Jakuc', className='mr-2'), 
                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:chris.jakuc@gmail.com'), 
                    html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/cjakuc/Energy-Demand-Forecast-Dayton'), 
                    html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/christopher-jakuc-30a887199/'), 
                    html.A(html.I(className='fab fa-twitter-square mr-1'), href='https://twitter.com/CJakuc'), 
                ], 
                className='lead'
            )
        )
    )
)

# Layout docs:
# html.Div: https://dash.plot.ly/getting-started
# dcc.Location: https://dash.plot.ly/dash-core-components/location
# dbc.Container: https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), 
    navbar, 
    dbc.Container(id='page-content', className='mt-4'), 
    html.Hr(), 
    footer
])


# URL Routing for Multi-Page Apps: https://dash.plot.ly/urls
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index
    elif pathname == '/predictions':
        return predictions.layout
    elif pathname == '/insights':
        return insights
    elif pathname == '/process':
        return process
    else:
        return dcc.Markdown('## Page not found')

# Use predictions_dict from predictions.py to create a new, static prediction function
# def predict(n_clicks, model, features):
#     if (n_clicks>=1):
#         if ('linear' in str(model)):
#             if ('short' in str(features)):
#                 # Linear short-term, all
#                 if ('all' in str(features)):
#                     return predictions_dict["linear short-all"]
#                 # Linear short-term, best
#                 elif ('best' in str(features)):
#                     return predictions_dict["linear short-best"]
#             elif ('long' in str(features)):
#                 # Linear long-term, all
#                 if ('all' in str(features)):
#                     return predictions_dict["linear long-all"]
#                 # Linear long-term, best
#                 elif ('best' in str(features)):
#                     return predictions_dict["linear long-best"]
#         elif ('linear' not in str(model)):
#             if ('short' in str(features)):
#                 # XGBoost short-term, all
#                 if ('all' in str(features)):
#                     return predictions_dict["xgboost short-all"]
#                 # XGBoost short-term, best
#                 elif ('best' in str(features)):
#                     return predictions_dict["xgboost short-best"]
#             elif ('long' in str(features)):
#                 # XGBoost long-term, all
#                 if ('all' in str(features)):
#                     return predictions_dict["xgboost long-all"]
#                 # XGboost long-term, best
#                 elif ('best' in str(features)):
#                     return predictions_dict["xgboost long-best"]
#     elif(n_clicks==0):
#         return ['Select a model and features to predict. Please allow a moment for everything to load after clicking predict.',
#                 plot_residuals(),
#                 # 'Here will be permutation importances.',
#                 aVp(y_pred['linear_unrealistic_all_train'])]

# Run app server: https://dash.plot.ly/getting-started
if __name__ == '__main__':
    app.run_server(debug=True)