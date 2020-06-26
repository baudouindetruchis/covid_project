"""
import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("dash_bootstrap_components")
install("plotly")
install("chart_studio")
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import chart_studio.plotly as py
import json
import requests
from collections import deque
import pytz
import mysql.connector
import sshtunnel
import time
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.express as px


def getAllLogs(camera):
    hostAddress = "Noptus.mysql.pythonanywhere-services.com"
    username = "Noptus"
    pwd = "794613852Video."
    database = "Noptus$CCTV_logs"
    sshtunnel.SSH_TIMEOUT = sshtunnel.TUNNEL_TIMEOUT = 5.0

    with sshtunnel.SSHTunnelForwarder(
            ('ssh.pythonanywhere.com'),
            ssh_username=username, ssh_password=pwd,
            remote_bind_address=(hostAddress, 3306)
    ) as tunnel:
        connection = mysql.connector.connect(
            user=username, password=pwd,
            host='127.0.0.1', port=tunnel.local_bind_port,
            database=database)
        connection.autocommit = True
        mycursor = connection.cursor()

        sql_log_query = "SELECT * FROM ML_Logs"+camera+";"
        mycursor.execute(sql_log_query)
        result = mycursor.fetchall()
        df = pd.DataFrame(result)

        mycursor.close()
        connection.close()
    tunnel.stop()
    return df

# Getting all the logs from the sshTunnel into a pandas dataframe
df = getAllLogs("")
df.columns = ['Log_ID','Datetime','Person_ID','Mask_B','Proximity_B','Gender_B','Age_I','Weight_I']
# setting datetime as the index
df.index = pd.to_datetime(df['Datetime'])
df1 = df

# computing the safety index for each entry
A = df.Age_I
M = df.Mask_B
W = df.Weight_I
P = df.Proximity_B
df["SafetyIndex"] = round( M*25 + (1-P)*25 + (1-abs(70-W)/100)*25 + (1-abs(20-A)/100)*25 )

# computing the averages
Overall_Average_Age = str(round(df["Age_I"].mean()))+"y"
Overall_Average_Weight = str(round(df["Weight_I"].mean()))+"kg"
Overall_Average_Gender = str(round(df["Gender_B"].mean()*100))+"% female"
Overall_Average_Proximity = str(round(df["Proximity_B"].mean()*100))+"%"
Overall_Average_Mask = str(round(df["Mask_B"].mean()*100))+"%"
Overall_Average_Safety = str(round(df["SafetyIndex"].mean()))+"%"


def BarGraphFromDataframe(X_axis,Y_axis,Title) :
    df["color"] = Y_axis.astype(str)
    Data = go.Bar(x=X_axis, y=Y_axis.count(),text=round(Y_axis,2),textposition='auto')

    Graph = dcc.Graph(
        figure={
            'data': [Data],
            'layout':go.Layout(title=Title)
        })
    return Graph

def ScatterGraphFromDataFrame(X_axis,Y_axis,Title,Y_title):
    df["color"] = Y_axis.astype(str)

    scale = [(0, "rgb(250,92,124)"),(1, "rgb(0,208,173)")]
    fig = px.scatter(df, x=X_axis, y=Y_axis, color=Y_axis,color_continuous_scale=scale,)
    fig.update_xaxes(
        rangeslider=dict(
            visible=True
        ),
        rangeselector=dict(
            buttons=list([
                dict(step="all"),
                dict(count=1, label=" Month ", step="month", stepmode="backward"),
                dict(count=7, label=" Week ", step="day", stepmode="backward"),
                dict(count=1, label=" Day ", step="day", stepmode="backward"),
                dict(count=1, label=" Hour ", step="hour", stepmode="backward")
            ])
        )
    )
    fig.update_yaxes(title_text=Y_title)

    return dcc.Graph(figure=fig)

def worldMap():
    df = px.data.gapminder().query("year == 2007")
    fig = px.scatter_geo(df, locations="iso_alpha",size="pop",projection="natural earth",scope="europe",height=370)
    return dcc.Graph(figure=fig)

# choosing the right metric
timeMetric=[]
timeMetric.append( [df.resample('10080T').mean(),"Week"] )
timeMetric.append( [df.resample('1440T').mean(),"Day"] )
timeMetric.append( [df.resample('60T').mean(),"Hour"] )
timeMetric.append( [df.resample('10T').mean(),"10 minutes interval"] )
timeMetric.append( [df.groupby(df.index.hour).mean(),"Hours of the day"] )
Metric = 3
df = timeMetric[Metric][0]
unit = timeMetric[Metric][1]

# generating the graphs
SafetyIndexGraph = ScatterGraphFromDataFrame(df.index,df.SafetyIndex,'Safety index','%')
Masks_Graph = ScatterGraphFromDataFrame(df.index,df.Mask_B*100,'Mask usage','%')
Proximity_Graph = ScatterGraphFromDataFrame(df.index,df.Proximity_B*100,'Proximity','%')
Age_Graph = ScatterGraphFromDataFrame(df.index,df.Age_I,'Age','Years old')
Weight_Graph = ScatterGraphFromDataFrame(df.index,df.Weight_I,'Weight','kg')
Gender_Graph = ScatterGraphFromDataFrame(df.index,df.Gender_B,'Gender',"%")

unit1 = "Gender_B"
unit2 = "Mask_B"
#Masks_Graph = BarGraphFromDataframe(df1[unit1],df1[unit2],unit2+' by '+unit1)

external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def getCountryInfo(country,index):
    json_url = "https://api.covid19api.com/live/country/"+country

    data = json.loads(requests.get(json_url).text   )
    if (country=="France") :
        latest_data = data[len(data)-2]
    else:
        latest_data = data[len(data)-1]

    output=[]
    output.append(str(latest_data['Confirmed']))
    output.append(str(latest_data['Deaths']))
    output.append(str(latest_data['Recovered']))
    output.append(str(latest_data['Active']))

    return output[index]

FranceCam = {
  "Link": "http://176.180.45.18:8082/mjpg/video.mjpg",
  "Name": "French Laundromat",
  "Country": "France",
  "City": "Beauvais",
  "Timezone":'Europe/Paris'

}
SwissCam = {
  "Link": "http://213.193.89.202/mjpg/video.mjpg",
  "Name": "Swiss street",
  "Country": "Switzerland",
  "City": "Himmelried",
  "Timezone":'Europe/Zurich'
}
SerbiaCam = {
  "Link": "http://93.87.72.254:8090/mjpg/video.mjpg",
  "Name": "Serbian Market",
  "Country": "Serbia",
  "City": "Novi Pazar",
  "Timezone":'Europe/Zurich'
}
cams = [SwissCam,FranceCam,SerbiaCam]
active = SerbiaCam

MainCard = dbc.Card(
        dbc.CardBody(
            [
                html.A([
                    html.Img(src="https://i.ibb.co/LtHfyfc/looking.png", width='100px', height="90px",
                             style={'float': 'left', "vertical-align": "middle"})
                ], href='covidwatchml.com')
                ,
                html.A([
                    html.Img(src="https://i.ibb.co/njcRQQf/garage.png",width='80px',height="80px",
                             style={'float':'right',"vertical-align":"middle"})
                ], href='https://www.linkedin.com/company/garageisep/')
                ,
                html.H1("Covid watch ML", className="card-title",
                        style={"font-size":"55px","padding-top": "12px","margin-left":"120px","vertical-align":"midle"}),

            ],
        ),
        style={"background-color":'rgb(41,50,63)',"color":'white'}
    )

statsRow1 = html.Tr([html.Td("Total infected"), html.Td(id="info1")])
statsRow2 = html.Tr([html.Td("Dead"), html.Td(id="info2")])
statsRow3 = html.Tr([html.Td("Recovered"), html.Td(id="info3")])
statsRow4 = html.Tr([html.Td("Active"), html.Td(id="info4")])
statsTableBody = [html.Tbody([statsRow1, statsRow2, statsRow3, statsRow4])]
statsTable = dbc.Table(statsTableBody, bordered=True)

CountryStatsCard = dbc.Card(
        dbc.CardBody([
                html.H1("Camera selection"),
                html.Br(),
                dcc.Dropdown(id='demo-dropdown',
                options=[
                    {'label': 'Swiss street', 'value': '0'},
                    {'label': 'France Laundromat', 'value': '1'},
                    {'label': 'Serbian Market', 'value': '2'},
                        ],
                value='2',style={"font-size":"15px"}
                ),
                html.Br(),
                html.H4(id="statsIntro"),
                html.Br(),
                statsTable,
                html.Br(),
            ],
        ),
        style={"height":'100%',"background-color":'white',"box-shadow":"0 0 6px 2px rgba(0,0,0,.1)","border-radius":"10px"}
    )

@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    active = cams[int(value)]
    return 'Current camera "{}"'.format(active["Name"])

@app.callback(
    dash.dependencies.Output('statsIntro', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    active = cams[int(value)]
    return 'COVID stats for '+active["Country"]

@app.callback(
    dash.dependencies.Output('video', 'src'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return cams[int(value)]["Link"]

@app.callback(
    dash.dependencies.Output('city', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_outpu(value):
    active = cams[int(value)]
    return 'Current city : '+(active["City"])




@app.callback(
    dash.dependencies.Output('info1', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output2(value):
    active = cams[int(value)]
    return getCountryInfo(active["Country"],0)

@app.callback(
    dash.dependencies.Output('info2', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output2(value):
    active = cams[int(value)]
    return getCountryInfo(active["Country"],1)

@app.callback(
    dash.dependencies.Output('info3', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output2(value):
    active = cams[int(value)]
    return getCountryInfo(active["Country"],2)

@app.callback(
    dash.dependencies.Output('info4', 'children'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output2(value):
    active = cams[int(value)]
    return getCountryInfo(active["Country"],3)


table_header = [
    html.Thead(html.Tr([html.Th("Characteristic"), html.Th("Evaluation"), html.Th("Confidence")]))
]
tablerow1 = html.Tr([html.Td("People"), html.Td("5"), html.Td("5%")])
tablerow2 = html.Tr([html.Td("Masked"), html.Td("20%"), html.Td("90%")])
tablerow3 = html.Tr([html.Td("Avg Weight"), html.Td("80 kg"), html.Td("20%")])
tablerow4 = html.Tr([html.Td("Avg age"), html.Td("50 years"), html.Td("12%")])
tablerow5 = html.Tr([html.Td("Proximity"), html.Td("20%"), html.Td("5%")])
tablerow6 = html.Tr([html.Td("Gender"), html.Td("60% female"), html.Td("5%")])

table_body = [html.Tbody([tablerow1, tablerow2, tablerow3, tablerow4,tablerow5,tablerow6])]
table = dbc.Table(table_header + table_body, bordered=True)

CurrentlyCard = dbc.Card(
    [
        dbc.CardHeader([    html.H2('Live on this camera '),
    ] ),
        dbc.CardBody(
            [
                table,
                html.A(html.Button('Refresh Data'),href='/'),
            ],
        ),
        ],
        style={"height":'100%',"background-color":'#fee8c8'}
    )

LiveVideoCard = dbc.Card(
        dbc.CardBody(
            [
                html.Img(id="video",src=active['Link'],width='500px',height="350px")
            ],
        ),
        style={"background-color":'white',"text-align":'center',"border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )



def GenerateGraphCard(CardTitle,width,graph,url) :

    margin_left=str(width)+'px'
    margin_left_text=str(width+10)+'px'

    topPart = html.Div([
            html.Img(src=url, width=margin_left, height="60px",
                     style={'float': 'left',"vertical-align": "middle"}),
            html.Div(CardTitle, className="card-title",
                     style={"margin-left": margin_left_text, "font-size": "35px",
                            "font-weight": "bold","color":"rgb(104,110,119)","vertical-align": "middle"}),
            #html.H3("Avg:59%", style={"color":"rgb(104,110,119)"}),
            ],style={'display':'inline-block;'})

    return dbc.Card(
        dbc.CardBody(
            [   topPart,
                html.Div([graph],style={"margin-top":"20px"})
            ],
        ),
        style={"background-color": "white","border-radius":"10px","box-shadow":"0 0 6px 2px rgba(0,0,0,.1)"}
    )

SafetyCard = GenerateGraphCard("Safety index",52,
                               SafetyIndexGraph,"https://i.ibb.co/Thh2kZh/safety.png")
MasksCard = GenerateGraphCard("Masks worn",90,
                              Masks_Graph,"https://i.ibb.co/VSZBdtq/mask.png")
AgeCard = GenerateGraphCard("Age",60,
                            Age_Graph,"https://i.ibb.co/gVp7sXj/age.png")
ProximityCard = GenerateGraphCard("Proximity",75,
                                  Proximity_Graph,"https://i.ibb.co/CzVMDt5/distance.png")
WeightCard = GenerateGraphCard("Weight",60,
                               Weight_Graph,"https://i.ibb.co/6sn3V7H/weight.png")
GenderCard = GenerateGraphCard("Gender",60,
                               Gender_Graph,"https://i.ibb.co/gmDNYkz/gender.png")


ContactCard = dbc.Card(
        dbc.CardBody(
            [
                html.Div([
                    html.Img(
                        src="https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fnoptus.pythonanywhere.com%2F",
                        alt="Hits", width=100, height=32),

                ],style={"padding":"5px","background-color":"white",'border-radius':"10px","width":"110"}),
                html.H2("Tips, suggestions ? Covidwatch@garageisep.com !", className="card-title"),
            ],
        ),
        style={"background-color":"rgb(0,208,173)","border-radius":"20px","color":"white"}
    )

map = html.Div([worldMap()],style={"box-shadow":"0 0 6px 2px rgba(0,0,0,.1)","border-radius":"20px"})
graphRow0 = dbc.Row([dbc.Col(id='card0', children=[MainCard], md=12)])
graphRow1 = dbc.Row([dbc.Col(id='card1', children=[CountryStatsCard], md=3), dbc.Col(id='card2', children=[LiveVideoCard], md=6), dbc.Col(id='video2', children=[map], md=3)],justify="around")
graphRow2 = dbc.Row([dbc.Col(id='card3', children=[SafetyCard], md=6),dbc.Col(id='card4', children=[MasksCard], md=6)])
graphRow3 = dbc.Row([dbc.Col(id='card5', children=[AgeCard], md=6),dbc.Col(id='card6', children=[ProximityCard], md=6)])
graphRow4 = dbc.Row([dbc.Col(id='card7', children=[WeightCard], md=6),dbc.Col(id='card8', children=[GenderCard], md=6)])
graphRow5 = dbc.Row([dbc.Col(id='card9', children=[ContactCard], md=12)])

Dashbody = html.Div([graphRow1,html.Br(),html.Br(),graphRow2,html.Br(),html.Br(),graphRow3,html.Br(),html.Br(),graphRow4,html.Br(),html.Br(),graphRow5,html.Br(),html.Br()],
                    style={'margin-left':'20px','margin-right':'20px'})
app.layout = html.Div([graphRow0,html.Br(),html.Br(),Dashbody,html.Br()],style={'backgroundColor':'rgb(244,245,249)'})

if __name__ == '__main__':
    app.run_server()