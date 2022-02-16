import sqlite3
import os
import csv

DB_FILEPATH = os.path.join(os.path.dirname(__file__), 'apart.db')

conn = sqlite3.connect(DB_FILEPATH)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS code")
cur.execute("DROP TABLE IF EXISTS apt_deal")
cur.execute("CREATE TABLE code(법정동코드 INTEGER PRIMARY KEY, 도시 VARCHAR, 시군구 VARCHAR);")
cur.execute("""CREATE TABLE apt_deal(id INTEGER PRIMARY KEY AUTOINCREMENT,
지역코드 INTEGER,
법정동 VARCHAR, 
거래일 DATE, 
아파트 VARCHAR, 
전용면적 INTEGER, 
층 INTEGER, 
건축년도 INTEGER, 
거래금액 INTEGER,
FOREIGN KEY(지역코드) REFERENCES code(법정동코드));""")

with open('data/Code Table.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cur.execute("INSERT INTO code(법정동코드, 도시, 시군구) VALUES(?, ?, ?);", (row['법정동코드'], row['도시'], row['시군구']))

with open('data/Apart Deal.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cur.execute("INSERT INTO apt_deal(지역코드, 법정동, 거래일, 아파트, 전용면적, 층, 건축년도, 거래금액) VALUES(? ,? ,?, ?, ?, ?, ?, ?);",
         (row['지역코드'], row['법정동'], row['거래일'], row['아파트'], row['전용면적'], row['층'], row['건축년도'], row['거래금액']))

conn.commit()
cur.close()
conn.close()