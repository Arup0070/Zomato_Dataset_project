from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from datetime import datetime
import datetime as d


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Delivery_person_Age=float(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = float(request.form.get('Vehicle_condition')),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries = request.form.get('multiple_deliveries'),
            Festival= request.form.get('Festival'),
            City = request.form.get('City')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=f"As per your Input ETA will be after {results} Minute arround {(datetime.now()+d.timedelta(minutes=results)).strftime('%I:%M %p')}")






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)

