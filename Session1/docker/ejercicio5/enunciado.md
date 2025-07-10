# Sirve un modelo de IA con FastAPI usando Docker

Objetivo: Dockerizar una API de predicción usando FastAPI y scikit-learn.

Estructura de archivos

fastapi-iris/
  main.py
  requirements.txt
  Dockerfile

1. main.py
```
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Entrena el modelo al iniciar el servidor
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier().fit(X, y)

class IrisFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: IrisFeatures):
    prediction = clf.predict([data.features])
    return {"prediction": int(prediction[0])}
```

2. requirements.txt

nginx
fastapi
uvicorn
scikit-learn

3. Dockerfile
```
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

4. Construir y ejecutar el contenedor
bash

docker build -t iris-fastapi .
docker run -p 8000:8000 iris-fastapi

5. Probar la API
Puedes usar curl o una herramienta como Postman para hacer una predicción.

Ejemplo usando curl:

bash

curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'


### Respuesta esperada:

{"prediction": 0}