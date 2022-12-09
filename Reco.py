# 1. Library imports
#import uvicorn
#from fastapi import Form, File, UploadFile, Request, FastAPI
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity

# 2. Create the app object
from flask import Flask, request
app = Flask(__name__)


def content_based(article_id,mtrx):
    similarity = cosine_similarity(mtrx,mtrx[article_id].reshape(1, -1))
    return (similarity.argsort(axis=0))[-6:-1]    

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/affiche', methods=["POST"])	
def affiche():
	return request.data


@app.route('/recom', methods=["POST"])
def recom():
	return _recom(request.data)



def _recom(data: str):
	articles_embeddings = pickle.load(open('Test.sav', 'rb'))
	id = int(data,base=10)
	reco = content_based(id,articles_embeddings)
	return ' '.join([str(tmp[0]) for tmp in reco])
	#return data
	#return pred


# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
#@app.get('/{name}')
#def get_name(name: str):
#    return {'message': f'Hello, {name}'}



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__=='__main__': 
    app.run(debug = True)