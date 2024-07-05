# %%
import requests
import json

# %% test hello world:

req = requests.get('http://127.0.0.1:5000/')
print(req.status_code)
print(req.headers)
print(req.content)

# %% test the listing model endpoint

req = requests.get('http://127.0.0.1:5000/models')
print(req.content)

# %% test simple url  templating
req = requests.get('http://127.0.0.1:5000/model/california_housing_0')
print(req.content)

# %% Requesting model details with an valid model name

req = requests.get('http://127.0.0.1:5000/model/details?modelname=california_housing_0')
print(req.json())

# %% Requesting model details with an invalid model name (error 404)

req = requests.get('http://127.0.0.1:5000/model/details?modelname=california_housig_0')
print(req.status_code)

# %% Request a prediction

data = {'longitude': -122.23,
        'latitude': 37.88,
        'housing_median_age': 41.0,
        'total_rooms': 880.0,
        'population': 322.0,
        'households': 126.0,
        'median_income': 8.3252}

req = requests.post('http://127.0.0.1:5000/predict/california_housing_0',
                headers = {'Content-Type': 'application/json'},
                data = json.dumps(data)
               )

print(req.json())

# %%