# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from joblib import load
import pandas as pd
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import eli5
from eli5.sklearn import PermutationImportance

# Imports from this application
from app import app

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
            "all" means that the model will use all available features while still following the 
            short-term versus long-term convention mentioned previously. To determine the 
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

# Load in all 8 pickled models
linear_unrealistic_all = load('assets/linear_unrealistic.joblib')
linear_realistic_all = load('assets/linear_realistic.joblib')
linear_unrealistic_best = load('assets/linear_unrealistic_best.joblib')
linear_realistic_best = load('assets/linear_realistic_best.joblib')
xgboost_unrealistic_all = load('assets/XGBoost_unrealistic.joblib')
xgboost_realistic_all = load('assets/XGBoost_realistic.joblib')
xgboost_unrealistic_best = load('assets/XGBoost_unrealistic_best.joblib')
xgboost_realistic_best = load('assets/XGBoost_realistic_best.joblib')
# Put pickled models in a dict
pickles = {
    'linear_unrealistic_all':linear_unrealistic_all,
    'linear_unrealistic_best':linear_unrealistic_best,
    'linear_realistic_all':linear_realistic_all,
    'linear_realistic_best':linear_realistic_best,
    'xgboost_unrealistic_all':xgboost_unrealistic_all,
    'xgboost_unrealistic_best':xgboost_unrealistic_best,
    'xgboost_realistic_all':xgboost_realistic_all,
    'xgboost_realistic_best':xgboost_realistic_best
    }

# Load in dataframe and do train, test split
data = pd.read_csv('https://github.com/cjakuc/DS-Unit2_Build-Week/blob/master/Data/BW2_wrangled%20(1).csv?raw=true', index_col=0)
train = data[data['year']<2014]
test = data[data['year']==2015]

# Encode data
onehot_encoder = ce.OneHotEncoder()
ordinal_encoder = ce.OrdinalEncoder()
linear_train = onehot_encoder.fit_transform(train)
linear_test = onehot_encoder.transform(test)
xgboost_train = ordinal_encoder.fit_transform(train)
xgboost_test = ordinal_encoder.transform(test)
# Put the encoded data in a dict
model_data = {
    'linear_train':linear_train, 'linear_test':linear_test,
    'xgboost_train':xgboost_train, 'xgboost_test':xgboost_test
}
# Save the features for each model
linear_unrealistic_all = linear_train.columns.tolist()
linear_unrealistic_best = linear_train.drop(columns=[
                        'season_1','season_4', 'MW',
                        'DailyAvgAirTemp',
                        'HourlyPrecipitation','season_2',
                        'HourlyWindDirection',
                        'year']).columns.tolist()
linear_realistic_all = linear_train.drop(columns=[
                        'MW_lag1','MW_lag24', 'MW']).columns.tolist()
linear_realistic_best = linear_train.drop(columns=[
                        'DailyAvgAirTemp_lag1', 'MW',
                        'MW_lag1',
                        'DailyCoolingDegreeDays_lag1',
                        'DailyCoolingDegreeDays',
                        'MW_lag24',
                        'year','HourlyWindDirection',
                        'season_3',
                        'DailyHeatingDegreeDays_lag365']).columns.tolist()
xgboost_unrealistic_all = xgboost_train.columns.tolist()
xgboost_unrealistic_best = xgboost_train.drop(columns=[
                        'year','DegUnder65', 'MW',
                        'DailyHeatingDegreeDays',
                        'DegOver65',
                        'DailyCoolingDegreeDays',
                        'DailyCoolingDegreeDays_lag1']).columns.tolist()
xgboost_realistic_all = xgboost_train.drop(columns=[
                        'MW_lag1','MW_lag24', 'MW']).columns.tolist()
xgboost_realistic_best = xgboost_train.drop(columns=[
                        'MW_lag1','DegUnder65', 'MW',
                        'DailyHeatingDegreeDays',
                        'DailyCoolingDegreeDays',
                        'DailyCoolingDegreeDays_lag1',
                        'MW_lag24','year',
                        'DegOver65']).columns.tolist()
# Create a list of features
features_list = [
    linear_unrealistic_all,
    linear_unrealistic_best,
    linear_realistic_all,
    linear_realistic_best,
    xgboost_unrealistic_all,
    xgboost_unrealistic_best,
    xgboost_realistic_all,
    xgboost_realistic_best
]

# Create X train, X test, y train, and y test for linear and XGBoost
X_train_linear = linear_train.drop(columns=['MW'])
X_train_xgboost = xgboost_train.drop(columns=['MW'])
X_test_linear = linear_test.drop(columns=['MW'])
X_test_xgboost = xgboost_test.drop(columns=['MW'])
y_train = train['MW']
y_test = test['MW']
# Create dict of X and y train and test
X_y_train_test = {
    'X_train_linear':X_train_linear,
    'X_train_xgboost':X_train_xgboost,
    'X_test_linear':X_test_linear,
    'X_test_xgboost':X_test_xgboost,
    'y_train':y_train,
    'y_test':y_test
}
# Create dict of y predictions for train and test
y_pred = {
    'linear_unrealistic_all_train':pickles['linear_unrealistic_all'].predict(X_train_linear),
    'linear_unrealistic_best_train':pickles['linear_unrealistic_best'].predict(X_train_linear[linear_unrealistic_best]),
    'linear_realistic_all_train':pickles['linear_realistic_all'].predict(X_train_linear[linear_realistic_all]),
    'linear_realistic_best_train':pickles['linear_realistic_best'].predict(X_train_linear[linear_realistic_best]),
    'xgboost_unrealistic_all_train':pickles['xgboost_unrealistic_all'].predict(X_train_xgboost),
    'xgboost_unrealistic_best_train':pickles['xgboost_unrealistic_best'].predict(X_train_xgboost[xgboost_unrealistic_best]),
    'xgboost_realistic_all_train':pickles['xgboost_realistic_all'].predict(X_train_xgboost[xgboost_realistic_all]),
    'xgboost_realistic_best_train':pickles['xgboost_realistic_best'].predict(X_train_xgboost[xgboost_realistic_best]),
    'linear_unrealistic_all_test':pickles['linear_unrealistic_all'].predict(X_test_linear),
    'linear_unrealistic_best_test':pickles['linear_unrealistic_best'].predict(X_test_linear[linear_unrealistic_best]),
    'linear_realistic_all_test':pickles['linear_realistic_all'].predict(X_test_linear[linear_realistic_all]),
    'linear_realistic_best_test':pickles['linear_realistic_best'].predict(X_test_linear[linear_realistic_best]),
    'xgboost_unrealistic_all_test':pickles['xgboost_unrealistic_all'].predict(X_test_xgboost),
    'xgboost_unrealistic_best_test':pickles['xgboost_unrealistic_best'].predict(X_test_xgboost[xgboost_unrealistic_best]),
    'xgboost_realistic_all_test':pickles['xgboost_realistic_all'].predict(X_test_xgboost[xgboost_realistic_all]),
    'xgboost_realistic_best_test':pickles['xgboost_realistic_best'].predict(X_test_xgboost[xgboost_realistic_best])
}

# # Load in unfit, pickled permuters
# linear_unrealistic_all_permuter = load('assets/linear_unrealistic_all_permuter.joblib')
# linear_realistic_all_permuter = load('assets/linear_realistic_all_permuter.joblib')
# linear_unrealistic_best_permuter = load('assets/linear_unrealistic_best_permuter.joblib')
# linear_realistic_best_permuter = load('assets/linear_realistic_best_permuter.joblib')
# xgboost_unrealistic_all_permuter = load('assets/XGBoost_unrealistic_all_permuter.joblib')
# xgboost_realistic_all_permuter = load('assets/XGBoost_realistic_all_permuter.joblib')
# xgboost_unrealistic_best_permuter = load('assets/XGBoost_unrealistic_best_permuter.joblib')
# xgboost_realistic_best_permuter = load('assets/XGBoost_realistic_best_permuter.joblib')
# Create a list of permuters
# permuters_list = [
#     linear_unrealistic_all_permuter,
#     linear_unrealistic_best_permuter,
#     linear_realistic_all_permuter,
#     linear_realistic_best_permuter,
#     xgboost_unrealistic_all_permuter,
#     xgboost_unrealistic_best_permuter,
#     xgboost_realistic_all_permuter,
#     xgboost_realistic_best_permuter
# ]
# # Fit the permuters
# for i in range(0,len(permuters_list)+1):
#     if (i < 4):
#         permuters_list[i] = permuters_list[i].fit(X_y_train_test['X_test_linear'][features_list[i]],X_y_train_test['y_test'])
#     else:
#         permuters_list[i] = permuters_list[i].fit(X_y_train_test['X_test_xgboost'][features_list[i]],X_y_train_test['y_test'])

# Create function to print out the train and test MAEs nicely
def to_pred_text(train_mae, test_mae, model, features):
    improvement = 301.48-test_mae
    imporvement_percent = 100*(improvement / 301.48)
    if ('un' in model):
        return f"""The {str(model).replace('_',' ').replace('unrealistic','short-term')} model with {features} features has a train Mean Absolute Error score of:  
                {train_mae:,.2f} MW.  
                It has a test Mean Absolute Error Score of:  
                {test_mae:,.2f} MW. An improvement on the baseline of {improvement:,.2f}, an imporvement of {imporvement_percent:,.2f} percent"""
    if ('un' not in model):
        return f"""The {str(model).replace('_',' ').replace('realistic','long-term')} model with {features} features has a train Mean Absolute Error score of:  
                {train_mae:,.2f} MW.  
                It has a test Mean Absolute Error Score of:  
                {test_mae:,.2f} MW. An improvement on the baseline of {improvement:,.2f}, an imporvement of {imporvement_percent:,.2f} percent"""

# Create a function to make a residuals plot
def plot_residuals(y_test=test['MW'], model=linear_unrealistic_all, y_pred=y_pred['linear_unrealistic_all_test'],default=True):
    if default == True:
        fig = px.scatter(x=pd.Series([0]), y=pd.Series([0]), opacity=0.3,
                        labels={'x':'Test Set Actual Values(MW)','y':'Residuals (MW)'},
                        template='plotly_dark')
        # Add a title
        title = {'xref':'paper', 'yref':'paper', 'x':0.0,'xanchor':'left',
                'yanchor':'bottom',
                'text':'Make a prediction',
                'font':dict(family='Arial',
                            size=20)}
        fig.update_layout(title=title)
        # Add horizontal red line
        h_line = []
        h_line.append(dict(
            type= 'line',
            yref= 'y', y0= 0, y1= 0,
            xref= 'paper', x0= 0, x1= 1,
            line=dict(color='red'),
            name='Residuals = 0'
            ))
        fig.update_layout(shapes=h_line)
        return fig
    residuals = (y_test - y_pred)
    fig = px.scatter(x=y_test, y=residuals, opacity=0.3,
                    labels={'x':'Test Set Actual Values(MW)','y':'Residuals (MW)'},
                    template='plotly_dark')
    # Add a title
    title = {'xref':'paper', 'yref':'paper', 'x':0.0,'xanchor':'left',
            'yanchor':'bottom',
            'text':'MW Prediction Residuals (2015)',
            'font':dict(family='Arial',
                        size=20)}
    fig.update_layout(title=title)
    # Add horizontal red line
    h_line = []
    h_line.append(dict(
        type= 'line',
        yref= 'y', y0= 0, y1= 0,
        xref= 'paper', x0= 0, x1= 1,
        line=dict(color='red'),
        name='Residuals = 0'
        ))
    fig.update_layout(shapes=h_line)
    return fig
fig = plot_residuals()

# Create a function to make the permutation importances
def p_importances(permuter,feature_names):
    permutation_importances = eli5.show_weights(
        permuter,
        top=None,
        feature_names=feature_names
    )
    return permutation_importances

# Create a function to make the actual vs predicted plot
def aVp(predicted, test=test, default=True):
    if default == True:
        df_ap = pd.DataFrame(data={'date':[1,50,100,150,200,250,300,350],'value':[0,0,0,0,0,0,0,0],'is_actual':[1,0,0,0,0,0,0,0]})
        fig = px.line(data_frame=df_ap, x='date', y='value', color='is_actual',
                    labels={'date':'Day of the Year',
                            'value':'MW',
                            'is_actual':'Is the actual value'},
                    template='plotly_dark')
        # Add a title
        title = {'xref':'paper', 'yref':'paper', 'x':0.0,'xanchor':'left',
                'yanchor':'bottom',
                'text':'Make a prediction',
                'font':dict(family='Arial',
                            size=20)}
        fig.update_layout(title=title)
        # Update x ticks
        season_x_labels=['January 1st',
                        'February 19th',
                        'April 10th',
                        'May 30th',
                        'July 19th',
                        'September 7th',
                        'October 27th',
                        'December 16th']
        season_x_ticks = [1,50,100,150,200,250,300,350]
        fig.update_xaxes(
            tickvals=season_x_ticks,
            ticktext=season_x_labels,
            tickangle=15
        )
        # Add footer
        annotations = []
        annotations.append(dict(xref='paper', yref='paper', x=-0.05, y=-0.25,
                                        xanchor='left', yanchor='bottom',
                                        text='Click on legend values to adjust figure',
                                        font=dict(family='Arial',
                                                size=12,
                                                color='gray'),
                                        showarrow=False))
        fig.update_layout(annotations=annotations)
        return fig
    # Make tidy dataframe w/ actual and predicted
    df_actual = pd.DataFrame(data={'date':test['date'],'value':test['MW'],'is_actual':True})
    df_predicted = pd.DataFrame(data={'date':test['date'],'value':predicted,'is_actual':False})
    df_ap = pd.concat([df_actual,df_predicted],ignore_index=True)
    # Make figure
    fig = px.line(data_frame=df_ap, x='date', y='value', color='is_actual',
                labels={'date':'Day of the Year',
                        'value':'MW',
                        'is_actual':'Is the actual value'},
                template='plotly_dark')
    # Add a title
    title = {'xref':'paper', 'yref':'paper', 'x':0.0,'xanchor':'left',
            'yanchor':'bottom',
            'text':'Actual MW vs Predicted MW (2015)',
            'font':dict(family='Arial',
                        size=20)}
    fig.update_layout(title=title)
    # Update x ticks
    season_x_labels=['January 1st',
                    'February 19th',
                    'April 10th',
                    'May 30th',
                    'July 19th',
                    'September 7th',
                    'October 27th',
                    'December 16th']
    season_x_ticks = [1,50,100,150,200,250,300,350]
    fig.update_xaxes(
        tickvals=season_x_ticks,
        ticktext=season_x_labels,
        tickangle=15
    )
    # Add footer
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=-0.05, y=-0.25,
                                    xanchor='left', yanchor='bottom',
                                    text='Click on legend values to adjust figure',
                                    font=dict(family='Arial',
                                            size=12,
                                            color='gray'),
                                    showarrow=False))
    fig.update_layout(annotations=annotations)
    return fig

fig1 = aVp(y_pred['linear_unrealistic_all_train'])

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

@app.callback(
    [Output('prediction-performance', 'children'),
    Output('residuals-figure','figure'),
    # Output('permutation-importances','children'),
    Output('actual-pred-plot', 'figure')],
    [Input('predictButton','n_clicks')],
    [State('model', 'value'), State('features', 'value')]
)
def predict(n_clicks, model, features, X_y_train_test=X_y_train_test, y_pred=y_pred, features_list=features_list):
    y_train = X_y_train_test['y_train']
    y_test = X_y_train_test['y_test']
    if (n_clicks>=1):
        if ('linear' in str(model)):
            if ('short' in str(features)):
                # Linear short-term, all
                if ('all' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_all_test'], default=False),
                            # p_importances(permuters_list[0],features_list[0]),
                            aVp(y_pred['linear_unrealistic_all_test'],default=False)]
                # Linear short-term, best
                elif ('best' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_best_test'], default=False),
                            # p_importances(permuters_list[1],features_list[1]),
                            aVp(y_pred['linear_unrealistic_best_test'],default=False)]
            elif ('long' in str(features)):
                # Linear long-term, all
                if ('all' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_all_test'], default=False),
                            # p_importances(permuters_list[2],features_list[2]),
                            aVp(y_pred['linear_realistic_all_test'],default=False)]
                # Linear long-term, best
                elif ('best' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_best_test'], default=False),
                            # p_importances(permuters_list[3],features_list[3]),
                            aVp(y_pred['linear_realistic_best_test'],default=False)]
        elif ('linear' not in str(model)):
            if ('all' in str(features)):
                # XGBoost short-term, all
                if ('all' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_all_test'], default=False),
                            # p_importances(permuters_list[4],features_list[4]),
                            aVp(y_pred['xgboost_unrealistic_all_test'],default=False)]
                # XGBoost short-term, best
                elif ('best' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_best_test'], default=False),
                            # p_importances(permuters_list[5],features_list[5]),
                            aVp(y_pred['xgboost_unrealistic_best_test'],default=False)]
            elif ('un' not in str(model)):
                # XGBoost long-term, all
                if ('all' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                    plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_realistic_all_test'], default=False),
                            # p_importances(permuters_list[6],features_list[6]),
                            aVp(y_pred['xgboost_realistic_all_test'],default=False)]
                # XGboost long-term, best
                elif ('best' in str(features)):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_realistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_realistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_realistic_best_test'], default=False),
                            # p_importances(permuters_list[7],features_list[7]),
                            aVp(y_pred['xgboost_realistic_best_test'],default=False)]
    elif(n_clicks==0):
        return ['Select a model and features to predict. Please allow a moment for everything to load after clicking predict.',
                plot_residuals(),
                # 'Here will be permutation importances.',
                aVp(y_pred['linear_unrealistic_all_train'])]

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

            It is also interesting to note that in figure 1 there appears to be a linear relationship amongst the 
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