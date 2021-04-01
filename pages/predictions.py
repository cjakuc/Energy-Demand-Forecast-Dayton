# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dill import dump, load

import os
import sys
sys.path.append(os.path.realpath('.'))
# Imports from this application
from app import app
from pages.pred_help import fig, fig1



# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## __**Predictions**__

            To predict the hourly energy demand for Dayton, Ohio in 2015, 
            choose a model type and a set of features. \n


            """
        ),
    ],
    md=4,
)

column1_5 = dbc.Col(
    [
        dcc.Markdown(
            """
            The two available model types are linear regression and XGBoost Regressor. The two 
            available sets of features are "all" and "best". As the name implies, 
            "all" means that the model will use all available features while still following a 
            short-term versus long-term convention that I will elaborate on shortly. To determine the 
            "best" features, I calculated the average permutation importances of the features in each 
            model over 5 iterations and selected only the features that improved MAE. You can learn more about 
            permutation importances [here](https://academic.oup.com/bioinformatics/article/26/10/1340/193348).
            \n
            The sets of features are further broken into short-term and long-term. 
            The short-term models make use of features that includes energy 
            data from as little as an hour before the desired prediction. 
            Outside of purely academic exploration, this type of prediction 
            could be useful for a utility company trying to quickly adjust 
            electricity output to hourly changes in demand. The long-term 
            models make use of features that include energy data from, at 
            minimum, a year before the desired predictions. This type of 
            forecast could be useful for anyone trying to estimate future 
            electricity demand, such as government entities dictating policy 
            or a utility company planning their future generation portfolio.
            \n
            Figures 1 and 2 are dynamically updated when you choose a model and a set of features then 
            press the predict button. They are also interactive and you can see the values for individual 
            observations by hovering with your cursor. For figure 2, be sure to click on the lines in the legend 
            to more easily visualize only the predicted or the actual values. You can also zoom on both figures by 
            selecting a section with your cursor. Double-clicking resets the plot.



            """
        ),
    ],
)

# Create custom CSS style for dropdown bars
custom_CSS_body = {
  'color': 'white',
  'background-color':'black' }

column2 = dbc.Col(
    [
        dcc.Markdown('### Choose the model type'),
        # Select model type dropdown
        dcc.Dropdown(
            id='model',
            options = [
                {'label': 'Linear Regression', 'value': 'linear'},
                {'label': 'XGBoost Regressor', 'value': 'xgboost'},
            ],
            value = 'linear',
            className='mb-4 text-dark',
        ),
        # Predict button
        html.Button(
            'Predict',
            id='predictButton',
            n_clicks=0),
    ],
    md=4,
)

column3 = dbc.Col(
    [
        dcc.Markdown('### Choose a Set of Features'),
        # Select features dropdown
        dcc.Dropdown(
            id='features',
            options =[
                {'label': 'Short-term: All', 'value': 'short all'},
                {'label': 'Short-term: Best', 'value': 'short best'},
                {'label': 'Long-term: All', 'value': 'long all'},
                {'label': 'Long-term: Best', 'value': 'long best'},
            ],
            value = 'short all',
            className='mb-4 text-dark',
            placeholder='short  all',
        ),
        # Reset button
        html.Button(
            'Reset',
            id='resetButton',
            n_clicks=0),
    ],
    md=4
)

column4 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Model Performance: 
            """
        ),
    ],
    md=4,
)

column5 = dbc.Col(
    [
        html.Div(id='prediction-performance')
    ]
)

# with open('assets/fig', 'rb') as f:
#     fig = load(f)
column6 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 1
            """
        ),
        dcc.Graph(id='residuals-figure', figure=fig,config=dict(autosizable=True))
    ],
    # md=5,
)

# column7 = dbc.Col(
#     [
#         html.Div(id='permutation-importances')
#     ],
#     md=5,
# )

# with open('assets/fig1', 'rb') as f:
#     fig1 = load(f)
column8 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 2
            """
        ),
        dcc.Graph(id='actual-pred-plot', figure=fig1,config=dict(autosizable=True))
    ],
    # md=5,
)

with open('assets/predictions_dict', 'rb') as f:
    predictions_dict = load(f)
with open('assets/blank_residuals', 'rb') as f:
    blank_residuals = load(f)
with open('assets/blank_avp', 'rb') as f:
    blank_avp = load(f)
# print(predictions_dict['linear short best'])
@app.callback(
    [Output('prediction-performance', 'children'),
    Output('residuals-figure','figure'),
    # Output('permutation-importances','children'),
    Output('actual-pred-plot', 'figure')],
    [Input('predictButton','n_clicks')],
    [State('model', 'value'), State('features', 'value')]
)
def predict(n_clicks, model, features):
    if (n_clicks>=1):
        # print(f"{model} {features}")
        return predictions_dict[f"{model} {features}"]
    elif(n_clicks==0):
        return ['Select a model and features to predict. Please allow a moment for everything to load after clicking predict.',
                blank_residuals,
                # 'Here will be permutation importances.',
                blank_avp]

@app.callback(
    Output('predictButton','n_clicks'),
    [Input('resetButton', 'n_clicks')]
)
def reset(resetButton):
    return 0

pdp_plot = html.Img(src='assets/Screen Shot 2020-03-05 at 10.13.47 PM.png',className='img-fluid')
shapley_plot = html.Img(src='assets/Screen Shot 2020-03-05 at 10.14.18 PM.png',className='img-fluid')
linear_p_imps = html.Img(src='assets/Screen Shot 2020-03-05 at 10.28.15 PM.png',className='img-fluid')
xgboost_p_imps = html.Img(src='assets/Screen Shot 2020-03-05 at 10.41.36 PM.png',className='img-fluid')
baseline_pred = html.Img(src='assets/Screen Shot 2020-03-05 at 11.00.48 PM.png',className='img-fluid')
rel_mw_hour = html.Img(src='assets/Screen Shot 2020-03-05 at 11.23.47 PM.png',className='img-fluid')

column9 = dbc.Col(
    [
        dcc.Markdown(
            f"""
            \n

            ## __**Insights**__

            \n
            """
               ),
    ],
    md=4,
)

column10 = dbc.Col(
    [
        dcc.Markdown(
            f"""
            \n

            In order to evaluate these models it is first helpful to compare them 
            to a baseline educated guess. For my baseline prediction I used 
            the mean of the target, MW. As can be seen by comparing 
            the predictions shown in figures 2 and 3, all of the models performed 
            much better than the baseline estimate.
            
            I elected to further evaluate the models with mean absolute 
            error (MAE). MAE is one of several methods used to compare 
            predictions of continuous variables with their actual outcomes. I chose MAE because 
            of its straightforward interpretibility, as it literally is the 
            average absolute difference between predicted and actual values. 
            The test MAE of the baseline model is 301.48 MW. If you use the 
            predict tool above you can see that all of the modeling options 
            that I employed significantly improve on this baseline. The short-term 
            models also perform significantly better than the long-term models 
            which is as expected. 

            It is also interesting to note that in figure 1 there appears to be a autocorrelation amongst the 
            residuals in the long-term prediction models, but not in the short-term prediction models. This suggests that 
            there might be potential benefit to be gained from further feature engineering or adding new data to the long-term 
            models.

            \n
            """
               ),
    ],
)

column11 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 3
            """
        ),
        baseline_pred
    ]
)

column12 = dbc.Col(
    [
        dcc.Markdown(
            f"""
            \n

            Now, let's take a look at the relationship between hourly MW values and a single feature, the hour of the day. 
            Below in figure 4, there is clearly a connection between MW and hour of the day. This relationship looks to be 
            non-monotonic and non-linear and this is further depicted in the figures on the index page. Non-monotonic 
            simply means that one variable does not continually increase, or continually decrease, as another variable increases.
            Therefore, it makes sense that we see in the permutation importances shown in figures 5 and 6 that 
            hour was more impactful in reducing MAE in the XGBoost model than the linear regression model. 
            This is because the random forest technique used in XGBoost easily allows for non-monotonic and non-linear effects. 
            Permutation importances simply represent the effect a feature has on predictions when the values for the features 
            are randomly shuffled (permuted). You can learn more about monotonicity [here](https://en.wikipedia.org/wiki/Monotonic_function) 
            and random forest modeling [here](https://en.wikipedia.org/wiki/Random_forest).  

            I also want to highlight that the two most impactful features shown in the permutation importances of the long-term 
            linear model are hourly sea level pressure and hourly station pressure. I posit that this becauase although these features 
            are not exactly equal, they are highly similar. The model is likely assigning them coefficients that have similar 
            sizes in opposite directions, with the difference incorporating noise. This is something I would like to explore further as 
            I continue to work on this web app.

            \n
            """
               ),
    ],
)

column13 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 4
            """
        ),
        rel_mw_hour
    ]
)

column14 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 5: Permutation Importances of the Long-Term Linear Model with All Features
            """
        ),
        linear_p_imps
    ]
)

column15 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 6: Permutation Importances of the Long-Term XGBoost Model with All Features
            """
        ),
        xgboost_p_imps
    ]
)

column16 = dbc.Col(
    [
        dcc.Markdown(
            f"""
            \n

            The last two figures can help us to further understand how the hour of the day effects individual 
            predictions. Figure 7 is a partial dependence plot for the Long-Term XGBoost model, 
            this represents how much a prediction changes as the hour changes. Figure 8 is a 
            shapley plot for the first observation in the Short-Term XGBoost model. In simple terms, a shapley plot 
            shows how much individual values effected a prediction. What is particularly interesting about 
            this plot is that it precisely depicts how impactful the previous hour's energy demand is to the prediction. 
            Hour is the second most impactful value and still has a strong influence on this prediction, but 
            it has less than half of the effect of the single hour lagged MW value.

            \n
            """
               ),
    ],
)


column17 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 7: Partial Dependence Plot for Hour of the Day in the Long-Term XGBoost Model
            """
        ),
        pdp_plot
    ]
)

column18 = dbc.Col(
    [
        dcc.Markdown(
            """
            #### Figure 8: Shapley Plot for the First Observation in the Short-Term XGBoost Model 
            """
        ),
        shapley_plot
    ]
)

column19 = dbc.Col(
    [
        dcc.Markdown(
            f"""
            \n
            ## __**Future Analysis**__

            In the future I would like to continue to develop this web app and continue to work on predicting energy demand. 
            One particular idea that I want to try out is looking at the effect of simulating future weather. With something 
            such as a Monte Carlo simulation I could utilize likely values of the future weather to make predictions with this 
            energy demand model and construct estimations for the most and least likely levels of energy demand. I would also 
            like to make a more general use model by incorporating other features such as population and information on 
            industries in the region.
            \n
            """
               ),
    ],
)


layout = html.Div(
    [
        dbc.Row([column1, column2, column3]),
        dbc.Row([column1_5]),
        dbc.Row([column4, column5]),
        dbc.Row([column6]),
        dbc.Row([column8]),
        dbc.Row([column9]),
        dbc.Row([column10]),
        dbc.Row([column11]),
        dbc.Row([column12]),
        dbc.Row([column13]),
        dbc.Row([column14]),
        dbc.Row([column15]),
        dbc.Row([column16]),
        dbc.Row([column17]),
        dbc.Row([column18]),
        dbc.Row([column19]),
    ]
)

# Pickle the output so it doesn't have to run everything dynamically
# with open("assets/predictions", "wb") as filename:
#     dump(layout, filename)

# Rewriting predict code to use it for cacheing all possibile callbacks
# def predict1(n_clicks, model, features, X_y_train_test=X_y_train_test, y_pred=y_pred, features_list=features_list):
# Deleted because it's long