import pandas as pd
import category_encoders as ce
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import eli5
from eli5.sklearn import PermutationImportance
from joblib import load
from dill import dump

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
                {test_mae:,.2f} MW. An improvement on the baseline of {improvement:,.2f}, an improvement of {imporvement_percent:,.2f} percent"""

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

def predict(n_clicks, model, features, X_y_train_test=X_y_train_test, y_pred=y_pred, features_list=features_list):
    y_train = X_y_train_test['y_train']
    y_test = X_y_train_test['y_test']
    if (n_clicks>=1):
        if ('linear' in str(model).lower()):
            if ('short' in str(features).lower()):
                # Linear short-term, all
                if ('all' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_all_test'], default=False),
                            # p_importances(permuters_list[0],features_list[0]),
                            aVp(y_pred['linear_unrealistic_all_test'],default=False)]
                # Linear short-term, best
                elif ('best' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_unrealistic_best_test'], default=False),
                            # p_importances(permuters_list[1],features_list[1]),
                            aVp(y_pred['linear_unrealistic_best_test'],default=False)]
            elif ('long' in str(features).lower()):
                # Linear long-term, all
                if ('all' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_all_test'], default=False),
                            # p_importances(permuters_list[2],features_list[2]),
                            aVp(y_pred['linear_realistic_all_test'],default=False)]
                # Linear long-term, best
                elif ('best' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['linear_realistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['linear_realistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['linear_realistic_best_test'], default=False),
                            # p_importances(permuters_list[3],features_list[3]),
                            aVp(y_pred['linear_realistic_best_test'],default=False)]
        elif ('linear' not in str(model).lower()):
            if ('short' in str(features).lower()):
                # XGBoost short-term, all
                if ('all' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_all_test'], default=False),
                            # p_importances(permuters_list[4],features_list[4]),
                            aVp(y_pred['xgboost_unrealistic_all_test'],default=False)]
                # XGBoost short-term, best
                elif ('best' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_unrealistic_best_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_unrealistic_best_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                            plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_unrealistic_best_test'], default=False),
                            # p_importances(permuters_list[5],features_list[5]),
                            aVp(y_pred['xgboost_unrealistic_best_test'],default=False)]
            elif ('long' in str(features).lower()):
                # XGBoost long-term, all
                if ('all' in str(features).lower()):
                    train_mae = mean_absolute_error(y_train,y_pred['xgboost_realistic_all_train'])
                    test_mae = mean_absolute_error(y_test,y_pred['xgboost_realistic_all_test'])
                    return [to_pred_text(train_mae, test_mae, model, features),
                    plot_residuals(X_y_train_test['y_test'],model,y_pred['xgboost_realistic_all_test'], default=False),
                            # p_importances(permuters_list[6],features_list[6]),
                            aVp(y_pred['xgboost_realistic_all_test'],default=False)]
                # XGboost long-term, best
                elif ('best' in str(features).lower()):
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

# Go through model_lst and feature_lst to create all possible combinations with predit(). Save to predictions_dict
# mod_lst = ["Linear Regression", "XGBoost Regressor"]
# feat_lst = ["Short-term: All", "Short-term: Best", "Long-term: All", "Long-term: Best"]
mod_lst = ["linear", "xgboost"]
feat_lst = ["short all", "short best", "long all", "long best"]
predictions_dict = {}
for mod in mod_lst:
    for feat in feat_lst:
        predictions_dict[f"{mod} {feat}"] = predict(n_clicks=1, model=mod, features=feat)

# Pickle outputs so it doesn't have to run everything dynamically
with open("assets/predictions_dict", "wb") as filename:
    dump(predictions_dict, filename)
with open("assets/blank_residuals", "wb") as filename:
    dump(plot_residuals(), filename)
with open("assets/blank_avp", "wb") as filename:
    dump(aVp(y_pred['linear_unrealistic_all_train']), filename)
# with open("assets/fig", "wb") as filename:
#     dump(fig, filename)
# with open("assets/fig1", "wb") as filename:
#     dump(fig1, filename)

# avp_blank = y_pred[aVp(y_pred['linear_unrealistic_all_train'])]