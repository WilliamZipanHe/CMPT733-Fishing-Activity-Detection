import sys
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np
from datetime import datetime as dt

import plotly.express as px
from plotly import graph_objs as go
from plotly.graph_objs import *

FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
#start dash
app = dash.Dash(
    __name__,
    external_stylesheets=[FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = '733 - Fishing Activity Detection'
server = app.server

#please keep this secret...env not used
mapbox_access_token = "pk.eyJ1IjoiZ2VudG9vIiwiYSI6ImNraWZzZjdsMzA4amIycXFwZTlnZmRvZWYifQ.zg_eiU9ZSpPwik7P8mC_Vg"

tab_style = {
    'borderTop': '1px solid black',
    'borderLeft': '0px',
    'borderRight': '0px',
    'borderBottom': '0px',
    'padding': '7px',
    'color': '#023047',
    'fontWeight': 'bold',
    'backgroundColor': 'lightgrey',
}

tab_selected_style = {
    'borderTop': '1px solid black',
    'borderLeft': '0px',
    'borderRight': '0px',
    'borderBottom': '0px',
    'backgroundColor': '#FFB703',
    'color': '#023047',
    'fontWeight': 'bold',
    'padding': '7px',
}

list_of_gear_type = [
    'trollers',
    'trawlers',
    'purse_seines',
    'pole_and_line',
    'fixed_gear',
    'drifting_longlines'
]

list_of_fishing = [
    'True',
    'False'
]

list_of_timespan = [
    'Past Day',
    'Past Week',
    'Past Month'
]

timespan_days_offset = {
    'Past Day': -1,
    'Past Week': -7,
    'Past Month': -30
}

marker_symbol = {
    1: 'star',
    0: 'circle'
}

#to run with parquet
data = pd.read_csv("data/dashboard_data.csv")
data['adjust_time_date'] = pd.to_datetime(data['adjust_time_date'])
data['flag_registry'] = data['flag_registry'].fillna("UNK")
data['is_fishing_color'] = data['is_fishing'].apply(lambda x: 'orange' if x==1 else 'blue')

pred = pd.read_csv("data/predictions_rl_v1.csv")
pred['mmsi'] = '[protected]'
pred['adjust_time_date'] = pd.to_datetime(pred['adjust_time_date'])
pred['pred_color'] = pred['pred'].apply(lambda x: 'orange' if x==1 else 'blue')

def recent_clicked_id():
    return dash.callback_context.triggered[0]['prop_id'].split('.')[0]

def hoverinfo_template(df):
    return "mmsi: " + df['mmsi'].astype("str") + '<br>' \
            + "lat: " + df['lat_x'].round(2).astype("str") + '<br>' \
            + "lon: " + df['lon_x'].round(2).astype("str") + '<br>' \
            + "time: " + df['adjust_time_date'].astype("str") + '<br>' \
            + "flag: " + df['flag_registry'] + '<br>' \
            + "length (m): " + df['length_m_registry'].round(2).astype("str") + '<br>' \
            + "tonnage (gt): " + df['tonnage_gt_registry'].round(2).astype("str") + '<br>' \
            + "engine power (kw): " + df['engine_power_kw_registry'].round(2).astype("str")

#general layout
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    className="eleven columns div-panel",
                    children=[
                        html.H1("Fishing Activity Detection"),
                    ],
                ),
                html.Div(
                    className="eleven columns div-graph",
                    children=[
                        dcc.Tabs([
                            dcc.Tab(
                                className='custom-tab',
                                label='Historical Data',
                                style=tab_style, selected_style=tab_selected_style,
                                children=[
                                    html.Div(
                                        className="div-banner",
                                        children=[
                                            html.Div(
                                                className="div-banner-box",
                                                children=['Click a vessel to view its cruising history.']
                                            ),
                                            html.Div(
                                                className="div-dropdown",
                                                children=[
                                                    dcc.Dropdown(
                                                        id="gear-type-dropdown",
                                                        options=[
                                                            {"label": j, "value": j}
                                                            for j in list_of_gear_type
                                                        ],
                                                        placeholder="Select vessel type",
                                                    ),

                                                ],
                                            ),
                                            html.Div(
                                                className="div-dropdown",
                                                children=[
                                                    dcc.Dropdown(
                                                        id="is-fishing-dropdown",
                                                        options=[
                                                            {"label": j, "value": j}
                                                            for j in list_of_fishing
                                                        ],
                                                        placeholder="Select fishing",
                                                    )
                                                ],
                                            ),
                                            html.Div(
                                                className="div-dropdown",
                                                children=[
                                                    dcc.Dropdown(
                                                        id="history-timespan-dropdown",
                                                        options=[
                                                            {"label": j, "value": j}
                                                            for j in list_of_timespan
                                                        ],
                                                        placeholder="Select timespan",
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="map-graph"),
                                ]),
                            dcc.Tab(
                                className='custom-tab',
                                label='Prediction',
                                style=tab_style,
                                selected_style=tab_selected_style,
                                children=[
                                    html.Div(
                                        className="div-banner",
                                        children=[
                                            html.Div(className="div-banner-box", children=['Predictions for a specific vessel near Vancouver in April, 2021.']),
                                        ]
                                    ),
                                    dcc.Graph(
                                        id="map-prediction",
                                        figure = go.Figure(
                                            data=[
                                                Scattermapbox(
                                                    lat=pred.lat_x,
                                                    lon=pred.lon_x,
                                                    customdata=pred.mmsi,
                                                    text = hoverinfo_template(pred),
                                                    hoverinfo="text",
                                                    mode="markers",
                                                    marker=dict(
                                                        color=pred.pred_color,
                                                    ),
                                                ),
                                            ],
                                            layout=Layout(
                                                uirevision=True,
                                                autosize=True,
                                                clickmode="event",
                                                margin=go.layout.Margin(l=0, r=35, t=0, b=0),
                                                showlegend=False,
                                                mapbox=dict(
                                                    accesstoken=mapbox_access_token,
                                                    center=dict(lat=48.65, lon=-126.04),
                                                    style="light",
                                                    bearing=0,
                                                    zoom=6,
                                                ),
                                            ),
                                        )
                                    ),
                                ])
                        ])
                    ],
                ),
            ],
        )
    ]
)

@app.callback(
    Output("map-graph", "figure"),
    [
        Input("gear-type-dropdown", "value"),
        Input("is-fishing-dropdown", "value"),
        Input("history-timespan-dropdown", "value"),
        Input('map-graph', 'clickData')
    ],
)
def update_graph(selectedGearType, selectedFishing, selectedTimespan, clickData):
    if selectedGearType and selectedGearType != 'all':
        data_temp = data[data["gear_type"] == selectedGearType]
    else:
        data_temp = data

    tf_dict = {"True": 1, "False": 0}
    if selectedFishing and selectedFishing != 'all':
        data_temp = data_temp[data_temp['is_fishing']==tf_dict[selectedFishing]]
    else:
        data_temp = data_temp

    # most recent point for each vessel
    data_temp = data_temp.loc[data_temp.groupby('mmsi').adjust_time_date.idxmax()]

    fig = go.Figure(
        data=[
            Scattermapbox(
                lat=data_temp.lat_x,
                lon=data_temp.lon_x,
                customdata=data_temp.mmsi,
                text = hoverinfo_template(data_temp),
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    color="#023047",
                    size=10
                ),
            ),
        ],
        layout=Layout(
            uirevision=True,
            autosize=True,
            clickmode="event",
            margin=go.layout.Margin(l=0, r=35, t=0, b=0),
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=45, lon=-19),
                style="light",
                bearing=0,
                zoom=1.5,
            ),
            updatemenus=[
                dict(buttons=([dict(
                                args=[
                                    {
                                        "mapbox.zoom": 1.5,
                                        "mapbox.center.lat": 45,
                                        "mapbox.center.lon": -19,
                                        "mapbox.bearing": 0,
                                        "mapbox.style": "light",
                                    }
                                ],
                                label="Reset Zoom",
                                method="relayout",
                            )
                        ]
                    ),
                    direction="left",
                    pad={"r": 0, "t": 0, "b": 0, "l": 0},
                    showactive=False,
                    type="buttons",
                    x=0.5,
                    y=0.05,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#4A3F3F",
                    borderwidth=1,
                    bordercolor="#b0b0b0",
                    font=dict(color="#b0b0b0"),
                )
            ],
        ),
    )

    if clickData and (recent_clicked_id()=="map-graph" or recent_clicked_id()=="history-timespan-dropdown"):
        clickDataHistory = data[data['mmsi']==clickData['points'][0]['customdata']]
        if selectedTimespan:
            clickDataHistory = clickDataHistory[clickDataHistory['adjust_time_date'] >= clickDataHistory['adjust_time_date'].max() + pd.DateOffset(days=timespan_days_offset[selectedTimespan]) ]
        hist_fig = Scattermapbox(
            lat=clickDataHistory.lat_x,
            lon=clickDataHistory.lon_x,
            customdata=clickDataHistory.mmsi,
            text=hoverinfo_template(clickDataHistory),
            hoverinfo="text",
            marker=dict(
                # color='#EE6C4D',
                color=clickDataHistory.is_fishing_color
            ),
        )
        fig.add_trace(hist_fig)

    fig.update_layout(hoverlabel_font_color='white', hoverlabel_bordercolor="lightgrey")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
