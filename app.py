import os
import numpy as np
import pandas as pd
import string
import warnings
import pickle
warnings.filterwarnings('ignore')
from flask import Flask, render_template, request, make_response, send_file
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn.model_selection import train_test_split

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
	return render_template('app.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'Erro - não foi encontrado o arquivo no upload'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return 'Erro - Selecione um arquivo válido'
        if file and allowed_file(file.filename):
            filename = file.filename
            df = validaCsv(file)
            name = geraLead(df)
            #filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return name
    return "Erro - o verifique se o arquivo do upload é um csv ou verifique o tamanho do arquivo" 

def validaCsv(fileUpload):
    df = pd.read_csv(fileUpload)
    return df

def treinaModelo(dfPopulacao, model):
    nr = NearMiss()
    X, y = nr.fit_sample(dfPopulacao.drop(['portifolio', 'id'], axis=1), dfPopulacao.portifolio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.01, stratify=y)
    model.fit(X_train, y_train)
    return model

def geraLead(dfPortifolio):
    modelo = pickle.load(open('modelo.sav', 'rb'))
    dfPop  = pd.read_csv('estaticos_market_tratado.csv')
    dfPop["portifolio"] = -1
    dfPop.loc[dfPop.id.isin(dfPortifolio.id).astype(int) > 0, "portifolio"] = 1
    modelo = treinaModelo(dfPop, modelo)
    dfPred = dfPop[dfPop["portifolio"] == -1]
    y_pred = modelo.predict(dfPred.drop(['portifolio', 'id'], axis=1))
    dfRetornoLeadsMaisAderentes = dfPred[y_pred == 1].copy()
    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    fileName = now.strftime("%S%M%H%d%m%Y")+".csv"
    dfRetornoLeadsMaisAderentes[["id"]].to_csv(fileName, index=False)
    return fileName

@app.route('/portifolio', methods=['POST']) 
def portifolio():
    idportifolio = request.form['idportifolio']
    df= pd.read_csv("estaticos_portfolio"+idportifolio+".csv")
    return geraLead(df)

@app.route('/download//<file>')  
def download(file):
    return send_file(file, as_attachment=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port) 