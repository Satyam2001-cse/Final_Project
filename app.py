import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
with open('model/breast_cancer_prediction_model.pkl','rb')as pickle_file:  
    model=pickle.load(pickle_file)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('breast_cancer_frontend.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
   input_features = [float(x) for x in request.form.values()]
   features_value = np.array([input_features])
   print(features_value)
  #mean_radius= request.form["mean radius"]  
   #mean_texture = request.form["mean_texture"]  
   #mean_smoothness = request.form["mean_smoothness"]  
   #mean_compactness = request.form["mean_compactness"]  
   #mean_symmetry = request.form["mean_symmetry"]  
   #mean_fractal_dimension = request.form["mean_fractal_dimension"]  
   #radius_error= request.form["radius_error"]  
   #textureerror = request.form["texture_error"]  
   #smoothness_error = request.form["smoothness_error"]  
   #compactness_error = request.form["compactness_error"]  
   #concave_points_error = request.form["concave_points_error"]  
   #symmetry_error = request.form["symmetry_error"]  
   
    
   features_name=['mean radius', 'mean texture', 'mean smoothness', 'mean compactness',
       'mean symmetry', 'mean fractal dimension', 'radius error',
       'texture error', 'smoothness error', 'compactness error',
       'concave points error', 'symmetry error']
    
   df = pd.DataFrame(features_value, columns=features_name)
   output = model.predict(df)
        
   if output == 0:
        res_val = "** breast cancer **"
   else:
        res_val = "no breast cancer"
        

   return render_template('breast_cancer_frontend.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)