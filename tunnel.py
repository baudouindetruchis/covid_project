from datetime import datetime
import mysql.connector
import sshtunnel
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

    sshtunnel.SSH_TIMEOUT = sshtunnel.TUNNEL_TIMEOUT = 5.0

    with sshtunnel.SSHTunnelForwarder(
        (PA_situation),
        ssh_username=username, ssh_password=pwd,
        remote_bind_address=(hostAddress, 3306),logger=logger,
    ) as tunnel:
        print("tunnel connected at port",tunnel.local_bind_port)
        connection = mysql.connector.connect(user=username, password=pwd,host='127.0.0.1',
                                             port=tunnel.local_bind_port,database=database,)
        connection.autocommit = True
        mycursor = connection.cursor()

        sql_log_command="INSERT INTO ML_Logs_"+camera+"(Datetime, Population_on_screen, Mask, Proximity, Gender, Age,Fitness) VALUES ({} ,{},{},{},{},{},{});".format(timestamp,Population,Mask,Proximity,Gender,Age,Fitness)
        mycursor.execute(sql_log_command)
        print(sql_log_command)
        print("Query executed !")

        mycursor.close()
        connection.commit()

        connection.close()

    tunnel.stop()
    print("tunnel disconnected")

if __name__ == "__main__":
    current = datetime.now().strftime('%Y%m%d%H%m%S')
    insertLog("Serbia2",current,42, "NULL", 0, "NULL", "NULL",0)
