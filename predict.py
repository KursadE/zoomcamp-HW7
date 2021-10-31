import pickle

model_file = 'tree_model_depth=10.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

#print('input', customer)
#print('lead', y_pred.round(4))


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

from flask import Flask, request, jsonify

app = Flask('lead')

@app.route('/predict', methods=['POST'])

def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    lead = prediction >= 0.5
    
    result = {
        'lead_probability': float(prediction),
        'lead': bool(lead),
    }

    return jsonify(result)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=9696, debug=True)
    except:
        pass