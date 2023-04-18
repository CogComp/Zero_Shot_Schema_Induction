#!python
import cherrypy
import cherrypy_cors
import json
import os
import argparse
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import torch

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

class MyWebService(object):

    @cherrypy.expose
    def index(self):
        return open('html/index.html', encoding='utf-8')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    
    def annotate(self):
        hasJSON = True
        result = {"status": "false"}
        try:
            # get input JSON
            data = cherrypy.request.json
        except:
            hasJSON = False
            result = {"error": "invalid input"}

        if hasJSON:
            result = predictor.predict(sentence=data['text'])
        return result
    
if __name__ == '__main__':
    print("")
    # INITIALIZE YOUR MODEL HERE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--port", default=6003, type=int, required=False,
                        help="port number to use")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device:', device)

    # IN ORDER TO KEEP IT IN MEMORY
    print("Starting rest service...")
    cherrypy_cors.install()
    config = {
        'global': {
            'server.socket_host': '127.0.0.1',
            'server.socket_port': args.port,
            'cors.expose.on': True
        },
        '/': {
            'tools.sessions.on': True,
            'cors.expose.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())

        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './html'
        },
        '/html': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './html',
            'tools.staticdir.index': 'index.html',
            'tools.gzip.on': True
        }
    }
    cherrypy.config.update(config)
    cherrypy.quickstart(MyWebService(), '/', config)
