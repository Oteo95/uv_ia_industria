# Cómo Desplegar una Aplicación de Machine Learning con FastAPI en Render y CI/CD usando GitHub Actions

### Lo que vamos a cubrir:

* Introducción al ML en producción y APIs

* Entrenamiento de un modelo de Machine Learning

*  Construcción de una aplicación FastAPI

*  Despliegue de la aplicación FastAPI

*  Automatización del despliegue con GitHub Actions

*  Conclusión


¡Vamos allá! 🚀


## 1. Introducción al Machine Learning en Producción y APIs

¿Qué significa poner un modelo en producción?

Desplegar un modelo implica integrarlo en un entorno donde otras personas puedan acceder a él, normalmente a través de una API. Esto requiere gestionar almacenamiento de datos, monitorización del rendimiento y asegurar la escalabilidad.


¿Qué es una API?

Una API (Interfaz de Programación de Aplicaciones) permite que dos aplicaciones se comuniquen entre sí. En ML, una API permite a los usuarios acceder a las predicciones de tu modelo sin necesidad de ver el código fuente.

¿Por qué usar FastAPI?

FastAPI es ideal para ML en producción porque es rápido, fácil de usar y soporta validación automática de datos, procesamiento asíncrono y documentación de API integrada. Facilita el escalado y la gestión eficiente de APIs.

Si quieres una visión global, consulta la documentación oficial de FastAPI.

## 2. Entrenamiento de un Modelo de Machine Learning

Utiliza y entrena el modelo a tu gusto

# 3. Construcción de una Aplicación FastAPI

### Paso 1: Prepara tu entorno


### Instala los paquetes necesarios
pip install fastapi uvicorn pydantic scikit-learn

Crea el archivo principal main.py:

```
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI(debug=True)

class SalaryPredictionRequest(BaseModel):
    country: str
    education_level: str
    years_of_experience: int

@app.get("/"")
def home():
    return {'text': 'Predicción de salario para desarrolladores de software'}

@app.get('/calculate_salary')
def predict_get(country: str, education_level: str, years_of_experience: int):

    output = 222222
    return {'El salario estimado es': output}

@app.post('/predict_salary')
def predict_post(request: SalaryPredictionRequest):
    output = 11111
    return {'El salario estimado es': output}

if __name__ == '__main__':
    uvicorn.run("mlfastapi:app", host="127.0.0.1", port=8000, reload=True)
```


### Paso 2: Ejecutar la aplicación

uvicorn mlfastapi:app --reload

Si visitas http://127.0.0.1:8000 verás el mensaje de bienvenida.

Accede a la documentación automática en:

Swagger UI: http://127.0.0.1:8000/docs

Redoc: http://127.0.0.1:8000/redoc

## 4. Desplegar la Aplicación FastAPI en Render
Desplegar una app FastAPI en Render es sencillo y tiene plan gratuito.

Pasos:

* Prepara tu app FastAPI

* Un requirements.txt con tus dependencias, incluyendo uvicorn.
```
fastapi
uvicorn
scikit-learn
pydantic
``` 

* Crea una cuenta en Render

* Click en New → Web Service.

* Conecta tu cuenta de GitHub y selecciona tu repositorio de FastAPI.

* Rama: Elige la que tenga tu app (generalmente main).

* Comando de Build: Render detecta y ejecuta automáticamente pip install -r requirements.txt.

* Comando de Start: Render usará uvicorn main:app --host 0.0.0.0 --port 8000.

* Elige el plan gratuito y haz clic en Create Web Service.

* Render construirá y desplegará tu app. Puedes seguir los logs desde el dashboard.

* Prueba la app desplegada. Visita https://tu-nombre-app.onrender.com

* Documentación interactiva: https://tu-nombre-app.onrender.com/docs

¡Y listo! Tu app FastAPI está online.

## 5. Automatiza el Despliegue con GitHub Actions
Automatizar el despliegue con GitHub Actions permite que cada cambio en tu repo de GitHub lance un despliegue automático en Render.

Pasos:
1. Crea la carpeta de workflows en tu repo:
```
mkdir -p .github/workflows
```
2. Crea el archivo deploy.yml:
```
name: Deploy FastAPI App

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r Session1/proyecto_final/requirements.txt

      - name: Run tests
        run: |
          # If you have any tests, add the command to run them here
          echo "Running tests"

      - name: Deploy to Render
        env:
          RENDER_API_KEY: ${{ secrets.RENDER_DEPLOY_URL  }}
        run: |
          curl -X POST $RENDER_API_KEY
```
Nota:

Cambia RENDER_DEPLOY_URL por el indicado en tu proyecto.


4. Agrega la RENDER_DEPLOY_URL de Render como secreto en GitHub

En GitHub: Ve a tu repo → Settings → Secrets and variables → Actions → New repository secret.

Nombre: RENDER_DEPLOY_URL

Valor: (tu RENDER_DEPLOY_URL)

5. Haz push del código a GitHub
```
git add .
git commit -m "Configurar despliegue automático con GitHub Actions"
git push origin main
```

6. Verifica el despliegue automático
GitHub Actions lanzará el workflow y, al finalizar, Render desplegará automáticamente tu app FastAPI.


7. Conclusión
En este ejercicio, aprendiste a desplegar un modelo de ML usando FastAPI y Render, además de configurar una canalización de CI/CD con GitHub Actions. 🚀 ¡Ahora tienes una API completamente funcional y disponible online! 🌐