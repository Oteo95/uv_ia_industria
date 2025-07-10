from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI(debug=True)


# Define a Pydantic model for the POST request body
class SalaryPredictionRequest(BaseModel):
    country: str
    education_level: str
    years_of_experience: int

@app.get('/')
def home():
    return {'text': 'Software Developer Salary Prediction'}

@app.get('/calculate_salary')
def predict_get(country: str, education_level: str, years_of_experience: int):
    output=31111
    return {'The estimated salary is {}'.format(output)}

@app.post('/predict_salary')
def predict_post(request: SalaryPredictionRequest):
    output = 100111
    return {'The estimated salary is {}'.format(output)}

if __name__ == '__main__':
    uvicorn.run("mlfastapi:app", host="127.0.0.1", port=8000, reload=True)