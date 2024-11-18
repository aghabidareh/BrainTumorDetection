from flask import Flask , render_template , request
import numpy as np
import os
from model import image_pre , predict

app = Flask(__name__)

UPLOAD_FOLDER = 'D:/projects/Data Science/BrainTumorDetection/app/static'
ALLOWED_EXTENTIOS = set(['jpg', 'jpeg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():  # put application's code here
    return render_template('index.html')

@app.route('/' , methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        file1.save(path)
        #return path
        data = image_pre(path)
        s = predict(data)
        if s == 1:
            result = 'No Brain Cancer'
        else:
            result = 'Brain Cancer'
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)
