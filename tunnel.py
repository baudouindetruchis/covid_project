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

    def insertLog(timestamp,Person_ID:int,Mask_B,Proximity_B,Gender_B,Age_I:int,Weight_I:int):
        sql_log_command="INSERT INTO ML_Logs(Datetime,Person_ID, Mask_B, Proximity_B, Gender_B, Age_I,Weight_I) VALUES ({},{},{},{},{},{},{});".format(timestamp,Person_ID,Mask_B,Proximity_B,Gender_B,Age_I,Weight_I)
        mycursor.execute(sql_log_command)

    def random_date(start, end):
        return start + timedelta(
            seconds=random.randint(0, int((end - start).total_seconds())),
        )

    current = datetime.now()
    old = datetime.now()-timedelta(days=10)
    randomTime = random_date(old,current)

    n=1000

    for i in range(n):
        randomTime = random_date(old, current).strftime('%Y%m%d%H%M%S')
        randomPersonID = random.randint(0,1000)
        randomMask = random.randint(0,1)
        randomProximity = random.randint(0,1)
        randomGender = random.randint(0,1)
        randomAge = random.randint(20,100)
        randomWeight = random.randint(20,100)
        insertLog(randomTime,randomPersonID, randomMask, randomProximity, randomGender, randomAge, randomWeight)
        print(str(i)+"/"+str(n))

    mycursor.close()
    connection.close()

tunnel.stop()
print("tunnel disconnected")