import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import json

import requests
import mysql.connector
import sshtunnel
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

print("Modules imported")

def getAllLogs(cameraCountry):
    hostAddress = "Noptus.mysql.pythonanywhere-services.com"
    username = "Noptus"
    pwd = "794613852Video."
    database = "Noptus$CCTV_logs"
    sshtunnel.SSH_TIMEOUT = sshtunnel.TUNNEL_TIMEOUT = 5.0
    with sshtunnel.SSHTunnelForwarder(
            ('ssh.pythonanywhere.com'),ssh_username=username, ssh_password=pwd,remote_bind_address=(hostAddress, 3306)
    ) as tunnel:
        connection = mysql.connector.connect(user=username, password=pwd,
            host='127.0.0.1', port=tunnel.local_bind_port,database=database)
        connection.autocommit = True
        mycursor = connection.cursor()

        sql_log_query = "SELECT * FROM ML_Logs_"+cameraCountry+";"
        mycursor.execute(sql_log_query)
        result = mycursor.fetchall()
        df = pd.DataFrame(result)

        mycursor.close()
        connection.commit()
        connection.close()
    tunnel.stop()

    df.columns = ['Log_ID', 'Datetime', 'Population', 'Mask_B', 'Proximity_B', 'Gender_B', 'Age_I', 'Weight_I']
    df.index = pd.to_datetime(df['Datetime'])
    df["Gender_B"]=0
    df["Mask_B"]=0
    df["Age_I"]=0
    A = df.Age_I
    M = df.Mask_B
    W = df.Weight_I
    P = df.Proximity_B

    df["SafetyIndex"] = round(M * 25 + (1 - P) * 25 + (1 - abs(70 - W) / 100) * 25 + (1 - abs(20 - A) / 100) * 25)
    return df

def getAverages(df):
    OverallAvg =[]
    OverallAvg.append(str(round(df["SafetyIndex"].mean())))
    OverallAvg.append(str(round(df["Mask_B"].mean()*100)))
    OverallAvg.append(str(round(df["Population"].mean())))
    OverallAvg.append(str(round(df["Proximity_B"].mean())))
    OverallAvg.append(str(round(df["Weight_I"].mean())))
    OverallAvg.append(str(round(df["Gender_B"].mean()*100)))
    return OverallAvg

def ScatterGraphFromDataFrame(X_axis,Y_axis,Y_title,colorScale):
    df["color"] = Y_axis.astype(str)
    fig = px.scatter(df, x=X_axis, y=Y_axis, color=Y_axis,color_continuous_scale=colorScale)
    fig.update_xaxes(rangeslider=dict(visible=True),rangeselector=dict(buttons=list([
        dict(step="all"),dict(count=1, label=" Month ", step="month", stepmode="backward"),
        dict(count=7, label=" Week ", step="day", stepmode="backward"),dict(count=1, label=" Day ", step="day", stepmode="backward"),
        dict(count=1, label=" Hour ", step="hour", stepmode="backward")]))
    )
    fig.update_yaxes(title_text=Y_title,)

    return fig

def worldMap(dfMap,active):
    lon = float(active['Lon'])
    lat = float(active['Lat'])
    fig = px.scatter_geo(dfMap,text='Camera',lon='Longitude',lat='Latitude',
                         projection="natural earth")
    fig.update_geos(showframe=False,showcountries=True,center=dict(lon=lon, lat=lat),
                    lataxis_range=[-10,10], lonaxis_range=[-20, 20])
    fig.update_layout(height=300, margin={"r": 10, "t": 0,"l": 10, "b": 0})
    return fig

df = getAllLogs("Serbia2")

print('Logs found')

# choosing the right metric
timeMetric=[]
timeMetric.append( [df.resample('10080T').mean(),"Week"] )
timeMetric.append( [df.resample('1440T').mean(),"Day"] )
timeMetric.append( [df.resample('60T').mean(),"Hour"] )
timeMetric.append( [df.resample('10T').mean(),"10 minutes interval"] )
timeMetric.append( [df.groupby(df.index.hour).mean(),"Hours of the day"] )
Metric = 3
df = timeMetric[Metric][0]

RedGreen = [(0, "rgb(250,92,124)"), (1, "rgb(0,208,173)")]
GreenRed = [(0, "rgb(0,208,173)"), (1, "rgb(250,92,124)")]
ContinuousPurple = [(0, "rgb(114,124,245)"), (1, "rgb(114,124,245)")]
Extremes = [(0, "rgb(250,92,124)"),(0.5, "rgb(0,208,173)"), (1, "rgb(250,92,124)")]
ExtremesGreen = [(0, "rgb(0,208,173)"),(0.5, "rgb(250,92,124)"), (1, "rgb(0,208,173)")]
colorPalettes = [RedGreen,GreenRed,ContinuousPurple,Extremes,ExtremesGreen]

SafetyIndexGraph = ScatterGraphFromDataFrame(df.index,df.SafetyIndex,'%',RedGreen)
Masks_Graph = ScatterGraphFromDataFrame(df.index,df.Mask_B*100,'%',RedGreen)
Proximity_Graph = ScatterGraphFromDataFrame(df.index,df.Proximity_B,'%',GreenRed)
Age_Graph = ScatterGraphFromDataFrame(df.index,df.Population,'Person detected',GreenRed)
Weight_Graph = ScatterGraphFromDataFrame(df.index,df.Weight_I,'%',RedGreen)
Gender_Graph = ScatterGraphFromDataFrame(df.index,df.Gender_B,"%",Extremes)

external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def getCountryInfo(country,abv):
    json_url = "https://api.covid19api.com/live/country/"+country
    json_url2 =  "https://restcountries.eu/rest/v2/alpha/"+abv

    dataCovid = json.loads(requests.get(json_url).text)
    dataCountry = json.loads(requests.get(json_url2).text)
    population = dataCountry['population']

    if (country=="France") :
        latest_data = dataCovid[len(dataCovid)-2]
    else:
        latest_data = dataCovid[len(dataCovid)-1]

    output=[]
    output.append(str(latest_data['Confirmed']))
    output.append(str(latest_data['Deaths']))
    output.append(str(latest_data['Recovered']))
    output.append(str(latest_data['Active']))
    output.append(population)
    perM = round(latest_data['Confirmed'] / (dataCountry['population']/1000),1)
    output.append(str(perM))
    mortality = round(latest_data['Deaths'] / dataCountry['population'],2)
    output.append(str(mortality))

    return output

CoordRows = [['Market',43.136670, 20.512220],
            ['laundromat',49.433330,2.083330],
            ['Street',47.421110,7.596590]]
colmns = ['Camera', 'Latitude', 'Longitude']
dfMap = pd.DataFrame(data=CoordRows, columns=colmns)

def getCamsList():
    json_url = "https://api.npoint.io/dc17bafef2764d5b3b9b"
    allInfos = json.loads(requests.get(json_url).text)
    camList = allInfos['cameras']
    return camList

def getCamInfo(name):
    camList = getCamsList()
    camInfo = camList[name]
    return camInfo

cams = getCamsList()
active = getCamInfo("Serbia1")

helperText = html.Div(
    html.Div([
        html.Div("Coming soon :"),
        html.Div("DataLab improvements, Loading animations, more cameras !!")]),
    style={"font-size": "20px", "background-color": "rgb(244,245,249)", "color": "rgb(104,110,119)",
           "border-radius": "10px", "padding": "8px", "text-align": "left",
           "float": "right", "vertical-align": "middle", "margin-top": "5px", "margin-right": "50px"}
)

MainCard = dbc.Card(
        dbc.CardBody(
            [
                html.A([
                    html.Img(src="https://i.ibb.co/LtHfyfc/looking.png", width='110px', height="90px",
                             style={'float': 'left', "vertical-align": "middle"})
                ], href='covidwatchml.com')
                ,
                html.A([
                    html.Img(src="https://i.ibb.co/njcRQQf/garage.png",width='80px',height="80px",
                             style={'float':'right',"vertical-align":"middle"})
                ], href='https://www.linkedin.com/company/garageisep/')
                ,helperText,
                html.H1("Covid watch ML", className="card-title",
                        style={"font-size":"50px","padding-top": "12px","margin-left":"120px","vertical-align":"midle"}),

            ],
        ),
        style={"background-color":'rgb(41,50,63)',"color":'white'}
    )

padding="10px"
statsRow0 = html.Tr([html.Td(id="statsIntro",style={"padding-left":padding}),
                     html.Td(id="info0")],style={"background-color":"rgb(244,245,249)"})
statsRow1 = html.Tr([html.Td(" Total infected",style={"padding-left":padding}),
                     html.Td(id="info1")])
statsRow2 = html.Tr([html.Td(" Dead",style={"padding-left":padding}),
                     html.Td(id="info2")],style={"background-color":"rgb(244,245,249)"})
statsRow3 = html.Tr([html.Td(" Recovered",style={"padding-left":padding}),
                     html.Td(id="info3")])
statsRow4 = html.Tr([html.Td(" Active",style={"padding-left":padding}),
                     html.Td(id="info4")],style={"background-color":"rgb(244,245,249)"})

statsTableBody = [html.Tbody([statsRow0,statsRow1, statsRow2, statsRow3, statsRow4],
                             style={"font-size":"15px","color":"rgb(104,110,119)"})]
statsTable = dbc.Table(statsTableBody)

topPartCamera = html.Div([
    html.Img(src="https://i.ibb.co/bFMBLvf/cctv.png", width = "53px", height = "35px",
             style = {'float': 'left',"vertical-align": "middle"}),
    html.Div("Camera selection",style={"font-weight":"bold","font-size":"30px","color":"rgb(104,110,119)"}),
    ], style = {'display': 'inline-block;'})


camList = getCamsList()
CameraSelection = dcc.Dropdown(id='demo-dropdown',
                options=[{'label': camList[k]["Name"], 'value': k} for k in camList.keys()],
                value='Serbia1',style={"font-size":"15px"},clearable=False
                )

TimeSelection = \
        dcc.RadioItems(id='radioTime',
    options=[
        {'label': ' Over time', 'value': 'time'},
        {'label': ' In a day', 'value': 'day'},
        {'label': ' By day', 'value': 'week'}
    ],
    value='time',labelStyle={'display': 'inline-block',"margin": "0 12px 0 12px"},style={"font-size":"15px"}
    )

CountryStatsCard = dbc.Card(
        dbc.CardBody([
                topPartCamera,html.Br(),
                CameraSelection,html.Br(),
                statsTable,html.Br(),
                TimeSelection, html.Br(),
        ],
        ),
        style={"height":'100%',"background-color":'white',"box-shadow":"0 0 6px 2px rgba(0,0,0,.1)","border-radius":"10px"}
    )


# Callback for country stats
@app.callback(
    [dash.dependencies.Output('video', 'src'),
     dash.dependencies.Output('statsIntro', 'children'),
     dash.dependencies.Output('info1', 'children'),
     dash.dependencies.Output('info0', 'children'),
     dash.dependencies.Output('info2', 'children'),
     dash.dependencies.Output('info3', 'children'),
     dash.dependencies.Output('info4', 'children'),
     dash.dependencies.Output('worldMap', 'figure'),
     dash.dependencies.Output("loading-output-1", "value"),
     dash.dependencies.Output("loading-output-2", "value")],
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    active = cams[value]

    infos = getCountryInfo(active["Country"],active["Abv"])
    updatedMap = worldMap(dfMap, active)
    updatedMap.update_layout(transition_duration=500)

    countryop = round(infos[4]/1000000,1)
    infected = infos[0]+" ("+str(round(int(infos[0])/infos[4],2))+"%)"

    return cams[value]["Link"],active["Country"]+" population",\
           infected,str(countryop)+'M',infos[1]+' ('+infos[5]+"%)",infos[2],infos[3],updatedMap,\
           value,value

# Callback for averages
@app.callback(
    [#dash.dependencies.Output('avg-target0', 'children'),
     #dash.dependencies.Output('avg-target1', 'children'),
     dash.dependencies.Output('avg-target2', 'children'),
     dash.dependencies.Output('avg-target3', 'children'),
     dash.dependencies.Output('avg-target4', 'children'),
     #dash.dependencies.Output('avg-target5', 'children'),
    ],
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    activeCam = cams[value]
    dbAdress = activeCam["db"]
    print(dbAdress)
    dfActive = getAllLogs(dbAdress)

    Averages = getAverages(dfActive)

    avgSafetyText = "Avg: "+str(Averages[0])+"%"
    avgMaskText = "Avg: "+str(Averages[1])+"%"
    avgAgeText = "Avg: "+str(Averages[2])+"/hour"
    avgProximityText = "Avg: "+str(Averages[3])+"%"
    avgWeightText = "Avg: "+str(Averages[4])+"%"
    avgGenderText = "Avg: "+str(Averages[5])+" female"

    return avgAgeText,avgProximityText,avgWeightText
    #return avgSafetyText,avgMaskText,avgAgeText,avgProximityText,avgWeightText,avgGenderText

# Callback for graphs
@app.callback(
    [#dash.dependencies.Output('cardGraph0', 'figure'),
     #dash.dependencies.Output('cardGraph1', 'figure'),
     dash.dependencies.Output('cardGraph2', 'figure'),
     dash.dependencies.Output('cardGraph3', 'figure'),
     dash.dependencies.Output('cardGraph4', 'figure'),
     #dash.dependencies.Output('cardGraph5', 'figure'),
     ],
    [dash.dependencies.Input('demo-dropdown', 'value'),
     dash.dependencies.Input('radioTime', 'value')])
def update_output(camera,time):
    activeCam = cams[camera]
    dbAdress = activeCam["db"]
    print(dbAdress)
    dfActive = getAllLogs(dbAdress)
    print("Yes")

    if(time =="time"):
        df = dfActive.resample('10T').mean()
    elif(time=="week"):
        df = dfActive.groupby(dfActive.index.day).mean()
    elif (time == "day"):
        df = dfActive.groupby(dfActive.index.hour).mean()

    SafetyIndexGraph = ScatterGraphFromDataFrame(df.index, df.SafetyIndex, '%', RedGreen)
    Masks_Graph = ScatterGraphFromDataFrame(df.index, df.Mask_B * 100, '%', RedGreen)
    Proximity_Graph = ScatterGraphFromDataFrame(df.index, df.Proximity_B , '%', GreenRed)
    Age_Graph = ScatterGraphFromDataFrame(df.index, df.Population, 'Person detected', GreenRed)
    Weight_Graph = ScatterGraphFromDataFrame(df.index, df.Weight_I, '%', RedGreen)
    Gender_Graph = ScatterGraphFromDataFrame(df.index, df.Gender_B, "%", Extremes)

    SafetyIndexGraph.update_layout(transition_duration=50)
    Masks_Graph.update_layout(transition_duration=50)
    Proximity_Graph.update_layout(transition_duration=50)
    Age_Graph.update_layout(transition_duration=50)
    Weight_Graph.update_layout(transition_duration=50)
    Gender_Graph.update_layout(transition_duration=50)

    #return SafetyIndexGraph,Masks_Graph,Age_Graph,Proximity_Graph,Weight_Graph,Gender_Graph
    return Age_Graph,Proximity_Graph,Weight_Graph


# Callback for Datalab
@app.callback(
    [dash.dependencies.Output('cardGraph6', 'figure'),
     dash.dependencies.Output("loading-output-datalab", "value")],
    [dash.dependencies.Input('LabDrop1', 'value'),
     dash.dependencies.Input('LabDrop2', 'value'),
     dash.dependencies.Input('LabDrop3', 'value'),
     dash.dependencies.Input('demo-dropdown', 'value'),
     dash.dependencies.Input('radioTime', 'value'),

     ])
def update_output(x,y,color,camera,time):
    activeCam = cams[camera]
    dbAdress = activeCam["db"]
    print(dbAdress)
    dfActive = getAllLogs(dbAdress)

    if(time =="time"):
        df = dfActive.resample('10T').mean()
    elif(time=="week"):
        df = dfActive.groupby(dfActive.index.day).mean()
    else:
        df = dfActive.groupby(dfActive.index.hour).mean()

    if(x=='index'):
        df_x = df.index
    else:
        df_x = df[x]

    if (y == 'index'):
        df_y = df.index
    else:
        df_y = df[y]

    datalab_graph = ScatterGraphFromDataFrame(df_x,df_y,"%",colorPalettes[color])
    datalab_graph.update_layout(transition_duration=50)
    datalab_graph.update_layout(xaxis_title=x,yaxis_title=y)

    if (x != 'index'):
        datalab_graph.update_xaxes(rangeslider=dict(visible=True))

    return datalab_graph,color



LiveVideoCard = dbc.Card(
        dbc.CardBody(
            [
            dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1"),style={"padding-top":"55%"}),
            html.Img(id="video",src=active['Link'],width='100%',height="350px"),
            ]
        ),
        style={"background-color":'white',"text-align":'center',
               "border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )

def GenerateGraphCard(CardTitle,width,graph,url,tooltip_id) :
    margin_left=str(width)+'px'
    margin_left_text=str(width+10)+'px'

    avgSection = html.Div(
        html.Div(id="avg-target"+str(tooltip_id)),
        style={"font-size":"20px","background-color":"rgb(244,245,249)","color":"rgb(104,110,119)",
               "border-radius":"5px","padding":"5px","width":"180px","text-align":"center",
               "float":"right","vertical-align":"middle","margin-top":"5px","margin-right":"5px"}
    )

    topPart = html.Div([
            avgSection,
            html.Img(src=url, width=margin_left, height="60px",id="tooltip-target"+str(tooltip_id),
                     style={'float': 'left',"vertical-align": "middle"}),

            html.Div(CardTitle, className="card-title",
                     style={"margin-left": margin_left_text, "font-size": "35px",
                            "font-weight": "bold","color":"rgb(104,110,119)","vertical-align": "middle"}),
            dbc.Tooltip(tooltipText[tooltip_id],
            target="tooltip-target"+str(tooltip_id),style={"font-size":"15px"}),
            ],style={'display':'inline-block;'})

    return dbc.Card(
        dbc.CardBody([topPart,
                html.Div([
                    dcc.Graph(figure=graph,id="cardGraph"+str(tooltip_id))
                ],style={"margin-top":"20px"})
            ],),
        style={"background-color": "white","border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )

def GenerateComingSoonGraphCard(CardTitle,width,graph,url,tooltip_id) :
    margin_left=str(width)+'px'
    margin_left_text=str(width+10)+'px'

    topPart = html.Div([
            html.Img(src=url, width=margin_left, height="60px",id="tooltip-target"+str(tooltip_id),
                     style={'float': 'left',"vertical-align": "middle"}),

            html.Div(CardTitle, className="card-title",
                     style={"margin-left": margin_left_text, "font-size": "35px",
                            "font-weight": "bold","color":"rgb(104,110,119)","vertical-align": "middle"}),
            dbc.Tooltip(tooltipText[tooltip_id],
            target="tooltip-target"+str(tooltip_id),style={"font-size":"15px"}),
            ],style={'display':'inline-block;'})

    return dbc.Card(
        dbc.CardBody([topPart,
                html.Div("Coming soon !",style={"font-size":"35px","margin-top":"200px",
                                              "margin-bottom":"200px","text-align":"center","color":"rgb(104,110,119)"}),
            ],),
        style={"background-color": "white","border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )

tooltipText=[]
tooltipText.append("We compute the safety score by factoring in : face masks worn, average age, average weight and distance between people")
tooltipText.append("A machine learning model estimates if a person is wearing a face mask or not")
tooltipText.append("A machine learning model estimates the age of each individual people on screen")
tooltipText.append("A model evaluates how close people are on screen")
tooltipText.append("A model evaluates the fitness of each individual on screen")
tooltipText.append("A model evaluates the gender of each individual on screen")
tooltipText.append("Choose your own parameters !")

SafetyCard = GenerateComingSoonGraphCard("Safety index",52,SafetyIndexGraph,"https://i.ibb.co/Thh2kZh/safety.png",0)
MasksCard = GenerateComingSoonGraphCard("Masks worn",90,Masks_Graph,"https://i.ibb.co/VSZBdtq/mask.png",1)
AgeCard = GenerateGraphCard("Population",60,Age_Graph,"https://i.ibb.co/pWXPvsk/population.png",2)
ProximityCard = GenerateGraphCard("Proximity",75,Proximity_Graph,"https://i.ibb.co/CzVMDt5/distance.png",3)
WeightCard = GenerateGraphCard("Fitness",60,Weight_Graph,"https://i.ibb.co/vPgJKc5/fitness.png",4)
GenderCard = GenerateComingSoonGraphCard("Gender",60,Gender_Graph,"https://i.ibb.co/gmDNYkz/gender.png",5)

def GenerateDataLabCard(graph):
    Parameters = html.Div(
        dbc.Row([
            dbc.Col(html.Div("X-axis"),md=1),
            dbc.Col(
                dcc.Dropdown(id='LabDrop1',
                         options=[
                            {'label': 'Time', 'value': "index"},
                            {'label': 'Safety', 'value': "SafetyIndex"},
                            {'label': 'Masks', 'value': "Mask_B"},
                            {'label': 'Fitness', 'value': "Weight_I"},
                            {'label': 'Population', 'value': "Population"},
                            {'label': 'Proximity', 'value': "Proximity_B"},
                            {'label': 'Gender', 'value': "Gender_B"},
                         ],
                         value="Population", style={"font-size": "15px","text-align":"left"}
                         ), md=3
                ),
            dbc.Col(html.Div("Y-axis"),md=1),
            dbc.Col(
                dcc.Dropdown(id='LabDrop2',
                         options=[
                             {'label': 'Safety', 'value': "SafetyIndex"},
                             {'label': 'Masks', 'value': "Mask_B"},
                             {'label': 'Fitness', 'value': "Weight_I"},
                             {'label': 'Population', 'value': "Population"},
                             {'label': 'Proximity', 'value': "Proximity_B"},
                             {'label': 'Gender', 'value': "Gender_B"},
                         ],
                         value= "Proximity_B", style={"font-size": "15px","text-align":"left"}
                         ),md=3
            ),
            dbc.Col(html.Div("Palette"),md=1),
            dbc.Col(
                dcc.Dropdown(id='LabDrop3',
                         options=[
                             {'label': 'Neutral', 'value': 2},
                             {'label': 'Extremes green', 'value': 4},
                             {'label': 'Extremes red', 'value': 3},
                             {'label': 'Top green', 'value': 1},
                             {'label': 'Top red', 'value': 0},
                         ],
                         value=3, style={"font-size": "15px","text-align":"left"}
                         ),md=3)
        ]
        ),
        style={"font-size":"20px","background-color":"rgb(244,245,249)","color":"rgb(104,110,119)",
               "border-radius":"5px","padding":"5px","width":"80%","text-align":"center",
               "float":"right","vertical-align":"middle","margin-top":"5px","margin-right":"5px"}
    )

    topPart = html.Div([
            Parameters,
            html.Img(src="https://i.ibb.co/LdCtKVb/datalab.png", width="60px", height="60px",id="tooltip-target6",
                     style={'float': 'left',"vertical-align": "middle"}),
            html.Div("DataLab", className="card-title",
                     style={"margin-left": "70px", "font-size": "35px",
                            "font-weight": "bold","color":"rgb(104,110,119)","vertical-align": "middle"}),
            dbc.Tooltip(tooltipText[6],
            target="tooltip-target6",style={"font-size":"15px"}),
            ],style={'display':'inline-block;'})

    return dbc.Card(
        dbc.CardBody([topPart,
                html.Div([
                    dcc.Loading(
                        id="loading-datalab",
                        type="default",
                        children=html.Div(id="loading-output-datalab"), style={"padding-top": "55%"}),
                    dcc.Graph(figure=graph,id="cardGraph6")
                ],style={"margin-top":"20px"})
            ],),
        style={"background-color": "white","border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )

DataLab_Graph = ScatterGraphFromDataFrame(df.Age_I,df.Weight_I,"%",Extremes)
DataLab_Graph.update_layout(
    xaxis_title="Age",
    yaxis_title="Masks")
DataLabCard = GenerateDataLabCard(DataLab_Graph)

ContactCard = html.Div([
            html.Div([
                html.Img(src="https://i.ibb.co/KNMqncx/Image3.png",style={"height":"100px","width":"100%"}),
                html.Img(src="https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fnoptus.pythonanywhere.com%2F",
                alt="Hits", width=120, height=32,style={"position":"absolute","top":"16px","left":"25px"}),
                html.H2("Tips, suggestions ? covidwatch@garageisep.com !",
                style={"position":"absolute", "bottom": "8px","left": "20px","color":"white","font-size":"25px"}),
                ],style={"float":"left","position":"relative"}),
            ],)

map = html.Div(dcc.Graph(figure=worldMap(dfMap,active),id="worldMap"))
mapCard = html.Div([
    dcc.Loading(
            id="loading-2",
            type="default",
            children=html.Div(id="loading-output-2"),style={"padding-top":"55%"}),
    html.Img(src="https://i.ibb.co/3rCfgFS/Image4.png",width="40px",height="40px",
             style={'float': 'left', "vertical-align": "middle","margin-top":"12px","margin-left":"12px"}),
    html.H1("Locations", className="card-title",
             style={"margin-left": "60px","padding-top":"15px",
                    "font-weight": "bold", "color": "rgb(104,110,119)", "vertical-align": "middle"}),
    map
],style={"height":'100%',"background-color":'white',
                                   "box-shadow":"0 0 6px 2px rgba(0,0,0,.1)","border-radius":"10px"})

graphRow0 = dbc.Row([dbc.Col(id='card0', children=[MainCard], md=12)])
graphRow1 = dbc.Row([dbc.Col(id='card1', children=[CountryStatsCard], md=3),
                     dbc.Col(id='card2', children=[LiveVideoCard], md=5),
                     dbc.Col(id='video2', children=[mapCard], md=4)],justify="around")
graphRow2 = dbc.Row([dbc.Col(id='card3', children=[SafetyCard], md=6),
                     dbc.Col(id='card4', children=[MasksCard], md=6)])
graphRow3 = dbc.Row([dbc.Col(id='card5', children=[AgeCard], md=6),
                     dbc.Col(id='card6', children=[ProximityCard], md=6)])
graphRow4 = dbc.Row([dbc.Col(id='card7', children=[WeightCard], md=6),
                     dbc.Col(id='card8', children=[GenderCard], md=6)])

graphRow5 = dbc.Row([dbc.Col(id='card9', children=[DataLabCard], md=12)])

graphRow6 = dbc.Row([dbc.Col(id='card10', children=[ContactCard], md=12)])

Dashbody = html.Div([graphRow1,html.Br(),html.Br(),graphRow2,html.Br(),html.Br(),graphRow3,html.Br(),html.Br(),graphRow4,html.Br(),html.Br(),graphRow5,html.Br(),graphRow6,html.Br()],
                    style={'margin-left':'20px','margin-right':'20px'})
app.layout = html.Div([graphRow0,html.Br(),html.Br(),Dashbody,html.Br()],style={'backgroundColor':'rgb(244,245,249)'})

if __name__ == '__main__':
    app.run_server()