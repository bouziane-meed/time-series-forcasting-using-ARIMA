
########################################################################
############# code : time series forecasting using ARIMA ###############
########################################################################

#importing libraries


import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.offline as py

import plotly.tools as tls
import plotly.graph_objects as go
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dash.dependencies import Input, Output, State

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = go.Figure()




# set the directory  of data
df = pd.read_excel("data/data.xlsx")







colors = {
    'background': '#ffffff',
    'text': '#f29807'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Div([ html.Img(src=app.get_asset_url('distr.png'), style={'width': '33%', 'float': 'left', 'display': 'inline-block'}),

########################################################################
########################################################################
    html.Div(children=[
    html.H1(
        children='Predictive analytics dashboards',
        style={
            'textAlign': 'center',
            'color': '#6593C8'
        }
    ),
    html.Div(children='Distra web application for sales data visualization and time series forecasting', style={
        'textAlign': 'center',
        'color': '#ffffff'
    })
    ], style={'width': '33%', 'display': 'inline-block', 'background': '#000000', 'border-radius': '10px', 'border': '2px solid red'}),
########################################################################
########################################################################
    html.Div(children=[
    html.H5(children='Enter a product name :'),
   
    html.Div(children=[html.Div(children=[dcc.Input(id='input-box', type='text', value= 'Article_32')], style={'float': 'left'}),
    html.Div(children=[html.Button('Submit', id='button')])]),
    html.H5(children='Select a distribution channel :'),

    dcc.Dropdown(
            id='canal-id',
            options=[
            {'label': 'Detail', 'value': 'Détail'},
            {'label': 'Distributor', 'value': 'Distributeur'},
            {'label': 'Wholesaler', 'value': 'Grossiste'},
            {'label': '1/2 Wholesaler', 'value': '1/2 Gros'},
            {'label': 'LMS', 'value': 'GMS'}
            ],
            value= 'Détail', style={'width': '91%'}
        ),
        ], style={'width': '27%', 'float': 'right'}
        )
        ]),
        html.Div(children=[
        html.H3(children=' .')
        ], style={'background': '#000000'}),
########################################################################
########################################################################

    html.Div([
    html.Div([
   
    dcc.Graph(id='graph-1'),
    dcc.Graph(id='graph-2'),
    ], style={'width': '50%', 'float': 'left'}),
########################################################################
########################################################################
    html.Div([
    
    dcc.Graph(id='graph-3'),
    dcc.Graph(id='graph-4'),
    dcc.Slider(
        id='future-slider',
        min=1,
        max=7,
        step=1,
        marks={
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7'
    },
        value=5,
    )

    ], style={'width': '48%', 'float': 'right'})
    ])

])

########################################################################
########################################################################











@app.callback(
    Output('graph-1', 'figure'),
    [Input('button', 'n_clicks'),Input('canal-id', 'value')], [State('input-box', 'value')])
def output_1(n_clicks,cnl,art):
    df_yearmonth = df
    df_yearmonth['Order Date'] = pd.to_datetime({'year':df_yearmonth['Annee'],'month':df_yearmonth['Mois'],'day':df_yearmonth['Jour']})
    df_yearmonth['date_YearMonth'] = df_yearmonth['Order Date'].dt.year.astype('str') + '-' + df_yearmonth['Order Date'].dt.month.astype('str') + '-01'
    df_yearmonth['date_YearMonth'] = pd.to_datetime(df_yearmonth['date_YearMonth'])
    dff = df.loc[df['code_article'] == art]
    dff = dff.loc[df['canal_principal'] == cnl]
    article_ym = dff.groupby(['date_YearMonth','code_article', 'canal_principal'])['CA_kMAD'].sum().reset_index()


    return {
        'data': [
                     go.Scatter(
            x=article_ym['date_YearMonth'],
            y=article_ym['CA_kMAD'],
         )
         ],
        'layout': {
            'title': 'Visualization of sales turnover per Month :',
            'xaxis' : {'title': 'Date'},
            'yaxis' : {'title': 'Turnover'}
        }
    }

########################################################################
########################################################################
@app.callback(
    Output('graph-2', 'figure'),
    [Input('button', 'n_clicks'),Input('canal-id', 'value')], [State('input-box', 'value')])
def output_2(n_clicks,cnl,art):
    df['Order Date'] = pd.to_datetime({'year':df['Annee'],'month':df['Mois'],'day':df['Jour']})
    dff = df.loc[df['code_article'] == art]
    dff = dff.loc[df['canal_principal'] == cnl]
    article = dff.groupby(['Order Date','code_article', 'canal_principal'])['CA_kMAD'].sum().reset_index()


    return {
        'data': [
            {'x': article['Order Date'], 'y': article['CA_kMAD'], 'type': 'category'},
            #{'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
        ],
        'layout': {
            'title': 'Visualization of sales turnover per Day :',
                'xaxis' : {'title': 'Date'},
            'yaxis' : {'title': 'Turnover'}

        }
    }


########################################################################
########################################################################
@app.callback(
    Output('graph-3', 'figure'),
   [Input('button', 'n_clicks'),Input('canal-id', 'value')], [State('input-box', 'value')])
def output_3(n_clicks,cnl,art):

    df['Order Date'] = pd.to_datetime({'year':df['Annee'],'month':df['Mois'],'day':df['Jour']})
    dff = df.loc[df['code_article'] == art]
    dff = dff.loc[df['canal_principal'] == cnl]
    article = dff.groupby(['Order Date','code_article', 'canal_principal'])['CA_kMAD'].sum().reset_index()


    article = article.groupby('Order Date')['CA_kMAD'].sum().reset_index()
    article = article.set_index('Order Date')
    article.index
    y = article['CA_kMAD'].resample('MS').sum()

    from pylab import rcParams
    rcParams['figure.figsize'] = 5, 5
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=True)
                results = mod.fit()
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)

    results = mod.fit()
    results.summary().tables[1]

    pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=True)
    pred_ci = pred.conf_int()



    return {
        'data': [go.Scatter(
            x=y['2018':].index,
            y=y['2018':],
            name='Actual'
        ),
        go.Scatter(
            x=pred.predicted_mean.index,
            y=pred.predicted_mean,
            name='Predicted'
        )],
        'layout': {
                'title': 'Sales turnover prediction per Month :',
                'xaxis' : {'title': 'Date'},
            'yaxis' : {'title': 'Turnover'}

            }
        }



########################################################################
########################################################################
@app.callback(
    Output('graph-4', 'figure'),
   [Input('button', 'n_clicks'),Input('canal-id', 'value'),Input('future-slider', 'value')], [State('input-box', 'value')])
def output_4(n_clicks,cnl,stp,art):

    df['Order Date'] = pd.to_datetime({'year':df['Annee'],'month':df['Mois'],'day':df['Jour']})
    dff = df.loc[df['code_article'] == art]
    dff = dff.loc[df['canal_principal'] == cnl]
    article = dff.groupby(['Order Date','code_article', 'canal_principal'])['CA_kMAD'].sum().reset_index()


    article = article.groupby('Order Date')['CA_kMAD'].sum().reset_index()
    article = article.set_index('Order Date')
    article.index
    y = article['CA_kMAD'].resample('MS').sum()

    from pylab import rcParams
    rcParams['figure.figsize'] = 5, 5
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=True)
                results = mod.fit()
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 0, 12),
                                enforce_stationarity=True,
                                enforce_invertibility=True)

    results = mod.fit()
    results.summary().tables[1]

    pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=True)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    y_truth = y['2019-01-01':]

    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    pred_uc = results.get_forecast(steps=stp)
    pred_ci = pred_uc.conf_int()



    return {
        'data': [go.Scatter(
        x=y.index,
        y=y,
        name='Actual'
    ),
    go.Scatter(
        x=pred_uc.predicted_mean.index,
        y=pred_uc.predicted_mean,
        name='Predicted'
    )],
        'layout': {
                'title': 'Sales turnover future prediction per Month :',
                'xaxis' : {'title': 'Date'},
            'yaxis' : {'title': 'Turnover'}

            }
        }

########################################################################
########################################################################







if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)









