import datetime
from datetime import timedelta,datetime
import mysql.connector
import sshtunnel
import pandas as pd
import random
import time

hostAddress = "Noptus.mysql.pythonanywhere-services.com"
username = "Noptus"
pwd = "794613852Video."
database = "Noptus$CCTV_logs"
PA_situation = 'ssh.pythonanywhere.com'

sshtunnel.SSH_TIMEOUT = sshtunnel.TUNNEL_TIMEOUT = 5.0

with sshtunnel.SSHTunnelForwarder(
    (PA_situation),
    ssh_username=username, ssh_password=pwd,
    remote_bind_address=(hostAddress, 3306)
) as tunnel:
    print("tunnel connected at port",tunnel.local_bind_port)
    connection = mysql.connector.connect(
    user=username, password=pwd,
    host='127.0.0.1', port=tunnel.local_bind_port,
    database=database)
    connection.autocommit = True
    mycursor = connection.cursor()

    def insertLog(camera, timestamp, Population, Mask, Proximity, Gender, Age, Fitness):
        sql_log_command="INSERT INTO ML_Logs_Serbia2(Datetime, Population_on_screen, Mask, Proximity, Gender, Age,Fitness) VALUES ({} ,{},{},{},{},{},{});".format(timestamp,Population,Mask,Proximity,Gender,Age,Fitness)
        mycursor.execute(sql_log_command)

    def random_date(start, end):
        return start + timedelta(
            seconds=random.randint(0, int((end - start).total_seconds())),
        )

    current = datetime.now().strftime('%Y%m%d%H')

    insertLog("Serbia2",current,10, "NULL", 0, "NULL", "NULL",0)

    mycursor.close()
    connection.close()

tunnel.stop()
print("tunnel disconnected")