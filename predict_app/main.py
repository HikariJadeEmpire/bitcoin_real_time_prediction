# Dash version : 2.10.2
# Plotly version : 5.9.0
# Pandas version : 2.0.3
# River version : 0.18.0

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import requests
import calendar
import datetime as dt
import math
import pandas as pd

from river import metrics, utils, compose, linear_model, preprocessing, optim, time_series

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
def mean(lst):
    return sum(lst) / len(lst)

fig = dict(
    data=[{'x': [], 'y': []}], 
    layout=dict(
        xaxis=dict(range=[]), 
        yaxis=dict(range=[])
        )
    )

extract_features = compose.TransformerUnion(
    get_ordinal_date,
    get_month_distances
)
model = (
    extract_features |
    time_series.SNARIMAX(
        p=5,
        d=0,
        q=0,
        m=12,
        sp=3,
        sq=6,
        regressor=(
            preprocessing.StandardScaler() |
            linear_model.LinearRegression(
                intercept_init=110,
                optimizer=optim.SGD(0.01),
                intercept_lr=0.3
            )
        )
    )
)

# model = preprocessing.TargetStandardScaler(regressor=model)

df = {
    'ETHUSDT':{'datetime':[], 'price':[], 'predicted':[]},
    'BTCUSDT':{'datetime':[], 'price':[], 'predicted':[]},
      }
av_rmse = []

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
                                           style={'font-size': 10,}
                                           ),
                        "value": "ETHUSDT",
                    },
                    {
                        "label": html.Span("BTC:USDT", 
                                           style={'font-size': 10,}
                                           ),
                        "value": "BTCUSDT",
                        "disabled":False,
                    }],
                    id='choice0',
                    style = {
                                'border-radius': '15px',
                             }
                    ),
        html.Br(), html.Br(),
        dcc.Input(id="input1", type="number", placeholder="number of seconds",
                  className="form-control form-control-sm", 
                  style={
                        'border-radius': '15px',
                        'textAlign': 'center',
                        }),
        html.Br(), html.Br(),
        html.Span("RMSE score will compute every 0 minutes 0 seconds", 
                    className="form-label mt-4",
                    id='text_compute',
                    style={'font-size': 12},
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
                            'font-size': 14
                            }
                        ),
                html.Div([dcc.Graph(id='graph', figure=fig),
                          dcc.Store(id='storage-df', storage_type='memory')
                          ], 
                          style={
                            'margin': '10px',
                            'border-radius': '10px',
                            }
                            ),
                dcc.Interval(id="interval", interval=1*1000, max_intervals=-1), # max_intervals = -1 : infinity, 0 : stop
                # html.Br(), # before details below the graph
                dbc.Row([
                dbc.Col([
                    dbc.Badge("RMSE", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled"),
                    html.Br(), html.Br(),
                    html.Span("current interval ( seconds )", 
                            className="form-label mt-4",
                            style={'font-size': 8}
                            ),
                    html.Br(), html.Br(),
                    dbc.Badge("count : 0", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled", id='main-interval'),
                        ], width = 2 ,
                            style={
                            'textAlign': 'center',
                            'align-items': 'center',
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            },
                        ),
                dbc.Col([
                    html.Br(),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                                    html.Div(
                                        [
                                            html.P("RMSE : 0", className="mb-1",id='rmse1',
                                                   style={'font-size': 12},
                                                   ),
                                            html.Small("status", className="text-success", id='qual0',
                                                       style={'font-size': 11},
                                                       ),
                                        ],
                                        className="d-flex w-100 justify-content-between",
                                    ),
                                    html.P("number of interval : 0", className="mb-1", id='n_itv1',
                                           style={'font-size': 12},
                                           ),
                                    html.Small("current score.", className="text-muted",
                                               style={'font-size': 12},
                                               ),
                                ]
                            ),
                        dbc.ListGroupItem([
                                    html.Div(
                                        [
                                            html.P("RMSE : 0", className="mb-1",id='rmse2',
                                                   style={'font-size': 12},
                                                   ),
                                            html.Small("status", className="text-success", id='qual1',
                                                       style={'font-size': 11},
                                                       ),
                                        ],
                                        className="d-flex w-100 justify-content-between",
                                    ),
                                    html.P("number of interval : 0", className="mb-1", id='n_itv2',
                                           style={'font-size': 12},
                                           ),
                                    html.Small("previous RMSE score.", className="text-muted",
                                               style={'font-size': 12},
                                               ),
                                ]
                            ),
                        ],
                        className="list-group-item list-group-item-action flex-column align-items-start active",
                        ),
                    html.Br(),
                    html.Small("The STATUS is shown by comparing the current score with the previous RMSE score.", className="form-label mt-4",
                                                       style={'font-size': 10},
                                                       ),
                        ], width = 3 ,
                            style={
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            },
                    ),
                dbc.Col([
                    dbc.Badge("Average RMSE", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled"),
                    html.Br(), html.Br(),
                    html.Span("computed average RMSE from the beginning", 
                            className="form-label mt-4",
                            style={'font-size': 10}
                            ),
                    html.Br(), html.Br(),
                    dbc.Badge("0", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled", id='av-rmse'),
                    html.Br(), html.Br(),
                    html.Small("status", className="text-success", id='av-rmse-q',
                                    style={'font-size': 11},
                                    ),
                    html.Br(),
                    html.Small("previous status", className="text-body-tertiary", id='av-rmse-q1',
                                    style={'font-size': 11},
                                    ),
                        ], width = 3 ,
                            style={
                            'textAlign': 'center',
                            'align-items': 'center',
                            'margin': '10px',
                            'border-radius': '10px',
                            # 'background-color':'rgb(255, 99, 71)',
                            },
                        ),
                dbc.Col([
                    html.Br(), html.Br(),
                    dbc.Badge("Recommendation", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary disabled"),
                    html.Br(), html.Br(),
                    html.Span("we recommend you to ...", 
                            className="form-label mt-4",
                            style={'font-size': 10}
                            ),
                    html.Br(), html.Br(),
                    dbc.Badge("HOLD", 
                                    #  color="white",
                                    #  text_color="warning" ,
                                     className="btn btn-secondary", id='decision'),
                    html.Br(), html.Br(),
                    html.Small("at : 0", className="text-body-tertiary", id='decision1',
                                    style={'font-size': 11},
                                    ),
                    html.Br(), html.Br(),
                    html.Small("or in the next 1 minutes from current RMSE interval", className="text-body-tertiary", id='decision2',
                                    style={'font-size': 11},
                                    ),
                        ], width = 2 ,
                            style={
                            'textAlign': 'center',
                            'align-items': 'center',
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
                # html.Br(), # after details below the graph
                dcc.Store(id='storage-info', storage_type='memory'),
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
    Output("text_compute", "children"), 
    [
        Input("input1", "value"),
    ]
)
def num_window(n):
    if n is None:
        raise dash.exceptions.PreventUpdate
    elif 0 <= n < ( 60 ) :
        windw = int( n )
        return "RMSE score will compute every {b} seconds".format(b=(windw%60))
    elif 0 <= n < ( 60*60 ) :
        windw = int( n )
        return "RMSE score will compute every {a} minutes {b} seconds".format(a=windw//60,b=(windw%60))
    elif ( 60*60 ) <= n :
        windw = int( n )
        return "RMSE score will compute every {c} hours {a} minutes {b} seconds".format(a=(windw%(60*60))//(60),
                                                                                        b=(windw%60),
                                                                                        c=(windw//(60*60)))
    else :
        raise dash.exceptions.PreventUpdate

@app.callback(
    Output("coin", "children"), [Input("choice0", "value")]
)
def coin_selection(n):
    if n is None:
        return "NONE"
    else:
        return f"{n}"
    
@app.callback(
    Output("cardhead", "children"),
    Output("main-interval", "children"),
    Output("storage-df", "data"),
    Output('interval', 'max_intervals'),
    Output("storage-info", "data"),
    [
        Input("choice0", "value"),
        Input('interval', 'n_intervals'),
        Input("input1", "value"),
     ]
)
def price_prediction(coin, interval, windw):

    metric = utils.Rolling(metrics.RMSE(), window_size=windw)
    
    if (windw is not None) and (windw != 0) and (coin is not None) :
        max_intervals = -1
        key = "https://api.binance.com/api/v3/ticker?symbol="+coin
        data = requests.get(key).json()

        price = float(data['lastPrice']) # price
        times = float(data['closeTime']) # timestamp

        model.learn_one(times, price)
        # y_pred = model.predict_one(times)

        pred = model.forecast(horizon=60)
        y_pred = ( pred )[0]

        if (100*price) <= (pred[-1]) :
            decision = 999999
        elif (pred[-1]) <= 0 :
            decision = -1
        else :
            decision = ( pred )[-1]

        sc_interval = 0
        rmse = 'x'

        my_datetime = dt.datetime.fromtimestamp(data['closeTime'] / 1000)

        if ((interval % windw) != 0 ) :
            df[coin]['datetime'].append(my_datetime)
            df[coin]['price'].append(price)
            df[coin]['predicted'].append(y_pred)

        elif ((interval % windw) == 0 ) :
            n_df = len(df[coin]['price'])
            for i in range(n_df-windw,) :
                # Update the error metric
                metric.update(df[coin]['price'][i], df[coin]['predicted'][i])
                rmse = float(str(metric)[5:].strip(' ').replace(',',''))

            sc_interval = interval
            
            df[coin]['datetime'].append(my_datetime)
            df[coin]['price'].append(price)
            df[coin]['predicted'].append(y_pred)
        
        info = [
            f'number of interval : {sc_interval}', 
            f'number of interval : {sc_interval-windw}', 
            rmse, 
            decision,
            price,
            f'number of interval : {sc_interval+60}',
                ]

    else :
        coin = '<COIN>'
        max_intervals = 0
        sc_interval = 0
        info = None
        windw = 0

    return f'  💸  Processing {coin} predictions in real time', f'count : {interval}', df , max_intervals, info

@app.callback(
    Output("graph", "figure"), 
    [
        Input('interval', 'max_intervals'), 
        Input('storage-df', 'data'),
        Input("choice0", "value"),
     ]
)
def plot_graph(max_intervals, df, coin):
    if ( (coin is None) or (df is None) ) or (max_intervals == 0):
        raise dash.exceptions.PreventUpdate
    elif (max_intervals == -1) :
        df = pd.DataFrame( data = df[coin] )
        if len(df) >= (16*60) :
            df = df.iloc[ (len(df)-(16*60)):,: ]
        elif ( 45 <= len(df) < (16*60) ) :
            df = df.iloc[ 15:,: ]

        df = [
            go.Scatter(
                x=df['datetime'],
                y=df['price'],
                name='price',
                mode= 'lines',
            ),
            go.Scatter(
                x=df['datetime'],
                y=df['predicted'],
                name='predicted',
                mode= 'lines'
            ),
            ]

    return {'data': df, "layout": {"title": {"text": coin}} }

@app.callback(
    Output("n_itv1", "children"), 
    Output("n_itv2", "children"),

    Output("rmse1", "children"),
    Output("rmse2", "children"),

    Output("qual0", "children"),
    Output("qual0", "className"),
    Output("qual1", "children"),
    Output("qual1", "className"),

    Output("av-rmse", "children"),
    Output("av-rmse-q", "children"),
    Output("av-rmse-q", "className"),

    Output("av-rmse-q1", "children"),
    Output("decision", "children"),
    Output("decision1", "children"),
    [
        Input('storage-info', 'data'),
        Input("rmse1", "children"),
        Input("qual0", "children"),
        Input("qual0", "className"),
        Input("av-rmse", "children"),
        Input("av-rmse-q", "children"),
     ]
)
def info_update(info, old_rmse, o_tag, o_tag_col, o_av_rmse, o_av_rmse_mssg):
    if (info is not None) and (info[2] != 'x') :
        rmse = info[2]
        decision = info[3]
        p_rmse = float(old_rmse[7:])
        o_av_rmse = float(o_av_rmse[:])

        av_rmse.append(rmse)
        
        if (len(av_rmse) > (60*60) ) :
            av_rmse = av_rmse[( len(av_rmse) - (60*60) ):]
            m_rmse = mean(av_rmse)
        elif (len(av_rmse) <= (60*60) ) :
            m_rmse = mean(av_rmse)

        if p_rmse < rmse : tag1 = '  performance drop!  '; tag1_color = 'text-danger'
        elif p_rmse > rmse : tag1 = '  performance gains!  '; tag1_color = 'text-success'
        elif p_rmse == rmse : tag1 = '  neutral..  '; tag1_color = 'text-info'

        if m_rmse < o_av_rmse : tag2 = f'  better! ( - {round(o_av_rmse-m_rmse,4):,} )  '; tag2_color = 'text-success'
        elif m_rmse > o_av_rmse : tag2 = f'  worse! ( + {round(m_rmse-o_av_rmse,4):,} )  '; tag2_color = 'text-danger'
        elif m_rmse == o_av_rmse : tag2 = '  neutral..  '; tag2_color = 'text-info'

        dcs = info[4]
        if decision > dcs : dcs = "BUY"
        elif decision == dcs : dcs = "HOLD"
        elif decision < dcs : dcs = "SELL"

        return info[0], info[1], f'RMSE : {rmse:.6f}', old_rmse, tag1, tag1_color, o_tag, o_tag_col, f"{m_rmse:.4f}", tag2, tag2_color, o_av_rmse_mssg, dcs, f"the price will meet : {round(decision,2):,} , at {info[5]}"
    else :
        raise dash.exceptions.PreventUpdate

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)