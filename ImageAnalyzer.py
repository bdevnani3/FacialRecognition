__author__ = 'Aaron'

import cv2
import pymssql

# connecting to the database
conn = pymssql.connect(server="BIGMAC\SQLEXPRESS", user="sa", password="password", database="test")

# a cursor is used to make queries
cursor = conn.cursor()

# executing the SQL query to get a face recognizer (XML) from the database
cursor.execute("SELECT Face_Recognizer FROM test.dbo.Drivers WHERE Driver_ID = 1")

#print the resulting XML File
line = cursor.fetchone()
while line:
    print line  
    line = cursor.fetchone()