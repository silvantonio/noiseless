from os import environ
import logging
import json
from flask import Flask, jsonify, Response, request
from flask import render_template
from machinelearning.ml_handler import GeneralHandler
from imagerecognition.ir_handler import GeneralIRHandler
import numpy as np

app = Flask(__name__)
file_handler = logging.FileHandler('app.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)


@app.route('/')
def index():
    app.logger.info('informing')
    app.logger.warning('warning')
    app.logger.error('screaming bloody murder!')
    return render_template('index.html', powered_by=environ.get('POWERED_BY', 'Antonio Silva'))


@app.route('/ml')
def ml():
    print(request.method)
    print(request.args.getlist('arg[]'))

    args = request.args.getlist('arg[]')

    gh = GeneralHandler()
    result = gh.knn.predict([args])

    data = {
        'result': int(result)
    }
    js = json.dumps(data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp

# @app.route('/mltest')
# def mltest():
#     gh = GeneralHandler()
#     data = gh.load_csv('data/listings/listings_test.csv')
#     target = gh.load_csv('data/listings/listings_test_target.csv')
#     gh.knn.test_train(data, target)
#
#     return render_template('index.html', powered_by=environ.get('POWERED_BY', 'Antonio Silva'))

@app.route('/ir')
def ir():
    #print(request.method)
    #print(request.args.getlist('arg[]'))
    #image = 'https://img.alicdn.com/bao/uploaded/i2/890482188/TB1viAXdAfb_uJkSnfoXXb_epXa_!!0-item_pic.jpg'
    image = request.args.getlist('url')
    girh = GeneralIRHandler()
    result = girh.predict(image_url=image)
    data = {
        'result': int(result)
    }
    js = json.dumps(data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
