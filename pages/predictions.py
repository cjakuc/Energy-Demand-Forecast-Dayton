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
        
            ## Predictions

            Your instructions: How to use your app to get new predictions.

            """
        ),
    ],
    md=4,
)

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

# Load in unfit, pickled permuters
linear_unrealistic_all_permuter = load('assets/linear_unrealistic_all_permuter.joblib')
linear_realistic_all_permuter = load('assets/linear_realistic_all_permuter.joblib')
linear_unrealistic_best_permuter = load('assets/linear_unrealistic_best_permuter.joblib')
linear_realistic_best_permuter = load('assets/linear_realistic_best_permuter.joblib')
xgboost_unrealistic_all_permuter = load('assets/XGBoost_unrealistic_all_permuter.joblib')
xgboost_realistic_all_permuter = load('assets/XGBoost_realistic_all_permuter.joblib')
xgboost_unrealistic_best_permuter = load('assets/XGBoost_unrealistic_best_permuter.joblib')
xgboost_realistic_best_permuter = load('assets/XGBoost_realistic_best_permuter.joblib')
# Create a list of permuters
permuters_list = [
    linear_unrealistic_all_permuter,
    linear_unrealistic_best_permuter,
    linear_realistic_all_permuter,
    linear_realistic_best_permuter,
    xgboost_unrealistic_all_permuter,
    xgboost_unrealistic_best_permuter,
    xgboost_realistic_all_permuter,
    xgboost_realistic_best_permuter
]
# # Fit the permuters
# for i in range(0,len(permuters_list)+1):
#     if (i < 4):
#         permuters_list[i] = permuters_list[i].fit(X_y_train_test['X_test_linear'][features_list[i]],X_y_train_test['y_test'])
#     else:
#         permuters_list[i] = permuters_list[i].fit(X_y_train_test['X_test_xgboost'][features_list[i]],X_y_train_test['y_test'])

# Create function to print out the train and test MAEs nicely
def to_pred_text(train_mae, test_mae, model, features):
    return f"""The {model} model with {features} features has a train Mean Absolute Error score of:  
            {train_mae}  
            And a test Mean Absolute Error Score of:  
            {test_mae} """

# Create a function to make a residuals plot
def plot_residuals(y_test=test['MW'], model=linear_unrealistic_all, y_pred=y_pred['linear_unrealistic_all_test']):
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
def aVp(predicted=y_pred['linear_unrealistic_all_test'], test=test):
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

fig1 = aVp()

column2 = dbc.Col(
    [
        dcc.Markdown('### Choose the model type'),
        # Select model type dropdown
        dcc.Dropdown(
            id='model',
            options = [
                {'label': 'Linear Regression: Unrealistic', 'value': 'linear_unrealistic'},
                {'label': 'Linear Regression: Realistic', 'value': 'linear_realistic'},
                {'label': 'XGBoost Regressor: Unrealistic', 'value': 'xgboost_unrealistic'},
                {'label': 'XGBoost Regressor: Realistic', 'value': 'xgboost_realistic'}
            ],
            value = 'linear_unrealistic',
            className='mb-4',
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
                {'label': 'All', 'value': 'all'},
                {'label': 'Best', 'value': 'best'},
            ],
            value = 'all',
            className='mb-4',
            placeholder='all',
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

            Text continued

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
def predict(n_clicks, model, features, X_y_train_test=X_y_train_test, y_pred=y_pred, permuters_list=permuters_list, features_list=features_list):
    y_train = X_y_train_test['y_train']
    y_test = X_y_train_test['y_test']
    if (n_clicks>=1):
        if ('linear' in str(model)):
            if ('un' in str(model)):
                # Linear unrealistic, all
                if (str(features) == 'all'):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_all_test']),
                            # p_importances(permuters_list[0],features_list[0]),
                            aVp(y_pred['linear_unrealistic_all_test'])]
                # Linear unrealistic, best
                elif (str(features) == 'best'):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_best_test']),
                            # p_importances(permuters_list[1],features_list[1]),
                            aVp(y_pred['linear_unrealistic_best_test'])]
            elif ('un' not in str(model)):
                # Linear realistic, all
                if (str(features) == 'all'):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_all_test']),
                            # p_importances(permuters_list[2],features_list[2]),
                            aVp(y_pred['linear_realistic_all_test'])]
                # Linear realistic, best
                elif (str(features) == 'best'):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_best_test']),
                            # p_importances(permuters_list[3],features_list[3]),
                            aVp(y_pred['linear_realistic_best_test'])]
        elif ('linear' not in str(model)):
            if ('un' in str(model)):
                # XGBoost unrealistic, all
                if (str(features) == 'all'):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_all_test']),
                            # p_importances(permuters_list[4],features_list[4]),
                            aVp(y_pred['xgboost_unrealistic_all_test'])]
                # XGBoost unrealistic, best
                elif (str(features) == 'best'):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_best_test']),
                            # p_importances(permuters_list[5],features_list[5]),
                            aVp(y_pred['xgboost_unrealistic_best_test'])]
            elif ('un' not in str(model)):
                # XGBoost realistic, all
                if (str(features) == 'all'):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                    plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_realistic_all_test']),
                            # p_importances(permuters_list[6],features_list[6]),
                            aVp(y_pred['xgboost_realistic_all_test'])]
                # XGboost realistic, best
                elif (str(features) == 'best'):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_realistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_realistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_realistic_best_test']),
                            # p_importances(permuters_list[7],features_list[7]),
                            aVp(y_pred['xgboostnrealistic_best_test'])]
    elif(n_clicks==0):
        return ['Select a model and features to predict. Please allow a moment for everything to load after clicking predict.',
                plot_residuals(),
                # 'Here will be permutation importances.',
                aVp()]

@app.callback(
    Output('predictButton','n_clicks'),
    [Input('resetButton', 'n_clicks')]
)
def reset(resetButton):
    return 0



layout = html.Div(
    [
        dbc.Row([column1, column2, column3]),
        dbc.Row([column4, column5]),
        dbc.Row([column6]),
        dbc.Row([column8])
    ]
)