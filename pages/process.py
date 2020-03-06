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
        
            # Process

            ### The Regression Problem
            Energy infrastructure is extremely critical for our modern life and standard of living. To maintain this infrastructure energy providers 
            must plan their generation portfolios carefully to ensure output always meets demand and prevent outages. 
            It is with this idea in mind that I set out to predict energy demand. Predicted energy demand allows these providers to successfully 
            and efficiently choose their investments. My target would include a variable in a unit of energy, in this cause megawatts (MW). 
            Going from there, I looked for matching time-series energy and weather data. I chose to use energy data from Dayton Power and Light 
            Company because what I found had an agreeable time scale and I assumed it as a small enough region for me to use weather data from 
            a single weather station when approximating the weather for the entire region. The weather data I used was from NOAA and was observed at 
            Dayton International Airport. One of the big challenges with the weather data was that it was sampled multiple times an hour, not on the hour. 
            To get it on the same hourly time scale as the energy data I used a resampling method and then linearly interpolated the small percentage of missing values. 
            All of my data cleaning can be found on my Github in Python notebooks located [here](https://github.com/cjakuc/DS-Unit2_Build-Week). I also 
            made a preliminary Medium blog post that goes over some of my data exploration 
            [here](https://medium.com/analytics-vidhya/visualizing-the-relationship-between-energy-demand-and-air-temperature-7d3f7de3ff0).  \n

            ### Techniques
            To predict the continuous variable MW, I chose to use both linear regression and XGBoost Regressor models. The linear model is a simpler, more 
            interpretible choice that still ended up farily accurate. My XGBoost model was built using the XGBoost package in Python and under the 
            hood it fits a random forest, then uses gradient boosting to tune the model. More on XGBoost Regressor can be found in its 
            documentation [here](https://xgboost.readthedocs.io/en/latest/python/python_api.html) or in a DataCamp tutorial on how to use it 
            [here](https://www.datacamp.com/community/tutorials/xgboost-in-python). All of the models were trained using data from 
            2005 through 2013, validated and tuned using data from 2014, and tested on data from 2015.
            My modeling was all intially done in the same Python notebooks where I did my data cleaning and processing. 
            I then used the joblib package to 'pickle' my models and import them into this web app that I created using VS code and 
            deployed with [Heroku](https://www.heroku.com/). All of my interactive figures were made using the plotly express package. 
            To get the permutation importances plots and shapley plot I used the eli5 package.  \n

            ### Leakage & Usefulness
            To prevent leakage I ensured that I never included that target variable, MW, when I trained my models. For the short-term versus long-term 
            scenario I considered leakage slightly differently. For the short-term models I included lagged features from as little as an hour before 
            the desired prediction. For the long-term models I included lagged features from, at minimum, a year before the desired predictions. This 
            was done with different use cases in mind. Short-term predictions could be used by utility companies to quickly adjust electrical output 
            by changing the status of portions of, or entire, peak production facilities. Long-term predictions could be used by government entities dictating energy policy 
            or utility companies considering investment in generation facilities. Still, the usefulness of all of these models is limited because of the way 
            I used the weather data. I used the actual weather observed at the time of my desired predictions. This was out of necessity and time-constraint but 
            in the future I would like to use simulations based on historical data to create a range of likely weather observations that I could then use to run 
            the models.


            """
        ),

    ],
)

layout = dbc.Row([column1])