from flask import Flask, jsonify, request
import os
import dotenv
import pandas as pd
from dill import load

def fullname(o):
  return o.__module__ + "." + o.__class__.__name__


dotenv.load_dotenv('.env')
model_path = os.getenv('MODEL_PATH')
app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify(greeting='hello', json=True), 201

@app.route('/models')
def list_models():
    list_files = os.listdir(model_path)
    if list_files:
        return jsonify(models=list_files)
    else:
        return "No Models", 404

@app.route('/model/<modelname>')
def show_model_name(modelname):
    return jsonify(modelname=modelname)

@app.route('/model/details')
def details():
    model = request.args.get('modelname')
    print(model)
    try:
        list_files = os.listdir(os.path.join(model_path,model))
        if list_files:
            models = list()
            for model_file in list_files:
                with open(os.path.join(model_path,model, model_file), "rb") as f:
                    pipeline_mlr = load(f)
                steps_obj = {"model_name": model_file ,"steps": []}
                for name, md in pipeline_mlr.steps:
                    steps_obj['steps'].append({
                        'name': name,
                        'class_name': fullname(md),
                    })
                models.append(steps_obj)
            return jsonify(steps_obj)
        else:
            return "No Models", 404
    except FileNotFoundError:
        return "No Models", 404


@app.route('/predict/<modelname>', methods=['POST'])
def predict(modelname):
    data = pd.DataFrame(request.get_json(), index=[0])
    try:
        list_files = os.listdir(os.path.join(model_path, modelname))
        if len(list_files) == 1:
            with open(os.path.join(model_path, modelname, list_files[0]), "rb") as f:
                pipeline_mlr = load(f)
            prediction = pipeline_mlr.predict(data)
            return jsonify(prediction.tolist())
        else:
            return "No Models", 404
    except FileNotFoundError:
        return "No Models", 404
    return jsonify(pred)

if __name__ == '__main__':
    app.run(debug=True)