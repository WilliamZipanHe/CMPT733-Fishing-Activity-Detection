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

# load data
data = pd.read_csv("data/dashboard_data.csv")
data['adjust_time_date'] = pd.to_datetime(data['adjust_time_date'])
data['flag_registry'] = data['flag_registry'].fillna("UNK")

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

def recent_clicked_id(context):
    return context.triggered[0]['prop_id'].split('.')[0]

def hoverinfo_template(df):
    return "mmsi: " + df['mmsi'].astype("str") + '<br>' \
            + '_______________________' + '<br>' \
            + "lat: " + df['lat_x'].round(2).astype("str") + '<br>' \
            + '_______________________' + '<br>' \
            + "lon: " + df['lon_x'].round(2).astype("str") + '<br>' \
            + '_______________________' + '<br>' \
            + "time: " + df['adjust_time_date'].astype("str") + '<br>' \
            + '_______________________' + '<br>' \
            + "flag: " + df['flag_registry']

#general layout
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="twelve columns div-panel",
                    children=[
                        # html.I(className="fab fa-yelp fa-3x", style={'display':'inline-block'}),
                        html.H2("Fishing Activity Detection"),
                        # html.H5("Browse Categories"),
                        html.Div(
                            className="row",
                            children=[
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
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="twelve columns div-graph",
                    children=[
                        dcc.Graph(id="map-graph"),
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
        Input('map-graph', 'clickData')
    ],
)
def update_graph(selectedGearType, selectedFishing, clickData):
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
                    colorscale= 'matter',
                    opacity=1,
                    sizemin=3,
                    cmin=1,
                    cmax=5,
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

    if clickData and recent_clicked_id(dash.callback_context)=="map-graph":
        clickDataHistory = data[data['mmsi']==clickData['points'][0]['customdata']]
        hist_fig = Scattermapbox(
            lat=clickDataHistory.lat_x,
            lon=clickDataHistory.lon_x,
            customdata=clickDataHistory.mmsi,
            text=hoverinfo_template(clickDataHistory),
            hoverinfo="text",
            marker=dict(
                color='#EE6C4D',
            ),
        )
        fig.add_trace(hist_fig)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
