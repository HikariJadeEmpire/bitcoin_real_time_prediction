import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go

import requests
import calendar
import datetime as dt
import math
import pandas as pd

from river import metrics, utils, compose, linear_model, preprocessing, optim

def get_month(x):
    return {
        calendar.month_name[month]: ( month == dt.datetime.fromtimestamp(x/1000.0).month ) for month in range(1, 13)
    }

def get_ordinal_date(x):
    return {'ordinal_date': dt.datetime.fromtimestamp(x/1000.0).toordinal()}

def get_month_distances(x):
    return {
        calendar.month_name[month]: math.exp(-(dt.datetime.fromtimestamp(x/1000.0).month - month) ** 2)
        for month in range(1, 13)
    }

fig = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[])
        )
    )
metric = utils.Rolling(metrics.RMSE(), window_size=600)
model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
        ('month', compose.FuncTransformer(get_month_distances)),
        )
    ),
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(0.02)
    ))
)

model = preprocessing.TargetStandardScaler(regressor=model)
df = {
    'ETHUSDT':{'datetime':[], 'price':[], 'predicted':[], 'rmse':[]},
    'BTCUSDT':{'datetime':[], 'price':[], 'predicted':[], 'rmse':[]},
      }

app = Dash(
    title='BITCOIN PRED',
    name='BITCOINPRED',
    use_pages=False,
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.MORPH]
)

app.layout = dbc.Container([
    dbc.Row([
    dbc.Col([
        html.Br(), html.Br(),
        html.Div(children=[html.H3(' 💰 Bitcoin prediction '), html.Br(), html.Br(),
                           dbc.Badge("NONE", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled", 
                                     id="coin")
                                     ], 
                           style = {'height':'200px','textAlign': 'center',}),
        html.Br(), html.Br(),
        html.Span("Select your coin", 
                    className="form-label mt-4",
                    style={'font-size': 12}
                    ),
        html.Br(), html.Br(),
        dcc.Dropdown([
                    {
                        "label": html.Span("ETH:USDT", 
                                           style={'font-size': 10, 
                                                  'padding-left': 10,
                                                  }
                                           ),
                        "value": "ETHUSDT",
                    },
                    {
                        "label": html.Span("BTC:USDT", 
                                           style={'font-size': 10, 
                                                  'padding-left': 10,
                                                  }
                                           ),
                        "value": "BTCUSDT",
                        "disabled":False,
                    }],
                    id='choice0',
                    style = {
                                'border-radius': '15px',
                                'textAlign': 'center',
                             }
                    ),
        html.Br(), html.Br(),
    ], 
    width = 2 ,
    style={
    'textAlign': 'center',
    'margin': '10px',
    'border-radius': '10px',
    # 'background-color':'rgb(255, 99, 71)',
    }
    ),
    dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.P(["The graph will be plot after you have select the coin"],
                        className="card-header", id='cardhead', 
                        style={
                            'textAlign': 'center',
                            'margin': '10px',
                            'border-radius': '10px',
                            }
                        ),
                html.Br(), html.Br(),
                html.Div([dcc.Graph(id='graph', figure=fig),
                          dcc.Store(id='storage-df', storage_type='memory')
                          ], 
                          style={
                            'margin': '10px',
                            'border-radius': '10px',
                            }
                            ),
                dcc.Interval(id="interval", interval=1*1000, max_intervals=-1), # max_intervals = -1 : infinity, 0 : stop
                html.Br(), html.Br(),
                dbc.Row([
                dbc.Col([
                    dbc.Badge("RMSE", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled"),
                    html.Br(), html.Br(),
                    html.Span("current interval", 
                            className="form-label mt-4",
                            style={'font-size': 12}
                            ),
                    html.Br(), html.Br(),
                    dbc.Badge("count : 0", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled", id='main-interval'),
                        ], width = 4 ,
                            style={
                            'textAlign': 'center',
                            'align-items': 'center',
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            },
                        ),
                dbc.Col([
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                                    html.Div(
                                        [
                                            html.P("RMSE : 0", className="mb-1",id='rmse1'),
                                            html.Small("Yay!", className="text-success", id='qual0'),
                                        ],
                                        className="d-flex w-100 justify-content-between",
                                    ),
                                    html.P("number of interval : 0", className="mb-1", id='n_itv1'),
                                    html.Small("latest.", className="text-muted"),
                                ]
                            ),
                        dbc.ListGroupItem([
                                    html.Div(
                                        [
                                            html.P("RMSE : 0", className="mb-1",id='rmse2'),
                                            html.Small("Yay!", className="text-success", id='qual1'),
                                        ],
                                        className="d-flex w-100 justify-content-between",
                                    ),
                                    html.P("number of interval : 0", className="mb-1", id='n_itv2'),
                                    html.Small("10 minutes ago.", className="text-muted"),
                                ]
                            ),
                        ],
                        className="list-group-item list-group-item-action flex-column align-items-start active",
                        )
                        ], width = 4 ,
                            style={
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            },
                    ),
                ], 
                    style={
                            'textAlign': 'center',
                            'align-items': 'center',
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            }
                ),
                html.Br(), html.Br(),
                ])
            ])
    ], 
    width = 9 ,
    style={
    'textAlign': 'center',
    'align-items': 'center',
    'margin': '10px',
    'border-radius': '10px',
    # 'background-color':'rgb(255, 99, 71)',
    }),
],
style={
    'textAlign': 'center',
    'align-items': 'center',
    'margin': '5px',
    'border-radius': '10px',
    }
    )
])



@app.callback(
    Output("coin", "children"), [Input("choice0", "value")]
)
def on_button_click(n):
    if n is None:
        return "NONE"
    else:
        return f"{n}"
    
@app.callback(
    Output("cardhead", "children"),
    Output("main-interval", "children"),
    Output("n_itv1", "children"),
    Output("n_itv2", "children"),
    Output("storage-df", "data"),
    Output('interval', 'max_intervals'),
    [
        Input("choice0", "value"),
        Input('interval', 'n_intervals'),
     ]
)
def on_button_click(coin, interval):
    
    if coin is None :
        max_intervals = 0
        # raise dash.exceptions.PreventUpdate
    else:
        max_intervals = -1
        key = "https://api.binance.com/api/v3/ticker?symbol="+coin
        data = requests.get(key).json()

        price = float(data['lastPrice']) # price
        times = float(data['closeTime']) # timestamp

        y_pred = model.predict_one(times)
        model.learn_one(times, price)

        # Update the error metric
        metric.update(price, y_pred)

        my_datetime = dt.datetime.fromtimestamp(data['closeTime'] / 1000)

        if (y_pred != 0) :
            df[coin]['datetime'].append(my_datetime)
            df[coin]['price'].append(price)
            df[coin]['predicted'].append(y_pred)
            df[coin]['rmse'].append(0)
        else : 
            # raise dash.exceptions.PreventUpdate
            pass

    return f'{coin}', f'count : {interval}', f'number of interval : {interval}', f'number of interval : {interval-1}', df , max_intervals

@app.callback(
    Output("graph", "figure"), 
    [
        Input('interval', 'max_intervals'), 
        Input('storage-df', 'data'),
        Input("choice0", "value"),
     ]
)
def on_button_click(max_intervals, df, coin):
    if ((coin is None) or (df is None)) or (max_intervals == 0):
        raise dash.exceptions.PreventUpdate
    elif (max_intervals == -1) :
        df = pd.DataFrame( data = df[coin] )

        df = [
            go.Scatter(
                x=df['datetime'],
                y=df['price'],
                name='price',
                mode= 'lines+markers',
            ),
            go.Scatter(
                x=df['datetime'],
                y=df['predicted'],
                name='predicted',
                mode= 'lines+markers'
            ),
            ]

    return {'data': df, "layout": {"title": {"text": coin}} }

if __name__ == '__main__':
    app.run(debug=True)