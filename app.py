import flask
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

model = None
with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return flask.render_template('index.html')
    
    if request.method == 'POST':
        state = request.form['state']
        city = request.form['city']
        date = int(datetime.today().strftime("%Y%m%d"))
        area = request.form['area']
        floor = request.form['floor']
        year = request.form['year']

        conn = sqlite3.connect('apart.db')
        cur = conn.cursor()
        cur.execute('SELECT 법정동코드 FROM code WHERE 도시=(?) AND 시군구=(?);',(state, city))
        code = cur.fetchone()[0]
        cur.execute('SELECT 법정동 FROM apt_deal WHERE 지역코드=(?) ORDER BY random() LIMIT 1;',(code,))
        dong = cur.fetchone()[0]
        cur.execute('SELECT 아파트 FROM apt_deal WHERE 법정동=(?) ORDER BY random() LIMIT 1;',(dong,))
        name = cur.fetchone()[0]
        cur.close()
        conn.close()

        df = pd.DataFrame(data = [[code, state, city, dong, date, name, area, floor, year]],
        columns = ['지역코드', '도시', '시군구', '법정동', '거래일', '아파트', '전용면적', '층', '건축년도'])
        pred = np.expm1(model.predict(df)[0])

        return render_template('index.html', pred = pred*10000)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5432)