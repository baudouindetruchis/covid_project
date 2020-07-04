import datetime
from datetime import timedelta,datetime
import sshtunnel
import pandas as pd
import random
import time
import logging
import pymysql


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def insertLog(camera, timestamp, Population, Mask, Proximity, Gender, Age, Fitness):
    hostAddress = "Noptus.mysql.pythonanywhere-services.com"
    username = "Noptus"
    pwd = "794613852Video."
    database = "Noptus$CCTV_logs"
    PA_situation = 'ssh.pythonanywhere.com'

    sshtunnel.SSH_TIMEOUT = sshtunnel.TUNNEL_TIMEOUT = 3.0

    with sshtunnel.SSHTunnelForwarder(
        (PA_situation),
        ssh_username=username, ssh_password=pwd,
        remote_bind_address=(hostAddress, 3306),logger=logger
    ) as tunnel:
        # print("tunnel connected at port",tunnel.local_bind_port)
        connection = pymysql.connect(
        user=username, password=pwd,
        host='127.0.0.1', port=tunnel.local_bind_port,
        database=database)
        mycursor = connection.cursor()

        sql_log_command="INSERT INTO ML_Logs_Serbia2(Datetime, Population_on_screen, Mask, Proximity, Gender, Age,Fitness) VALUES ({} ,{},{},{},{},{},{});".format(timestamp,Population,Mask,Proximity,Gender,Age,Fitness)
        mycursor.execute(sql_log_command)
        connection.commit()

        mycursor.close()
        connection.close()

    tunnel.stop()
    # print("tunnel disconnected")

if __name__ == "__main__":
    current = datetime.now().strftime('%Y%m%d%H%m%S')
    insertLog("Serbia2",current,123456789, "NULL", 0, "NULL", "NULL",0)
