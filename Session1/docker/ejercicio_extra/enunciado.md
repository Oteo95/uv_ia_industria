# Ejercicio extra: Utiliza Docker como Devcontainer en VS Code

Objetivo: Aprender a usar Docker para crear entornos de desarrollo reproducibles con devcontainer en Visual Studio Code.

Solución paso a paso
Instala la extensión "Dev Containers" en VS Code.

Estructura de ejemplo para un proyecto de IA (Python):

.devcontainer/
  devcontainer.json
  Dockerfile
requirements.txt
main.py

# Ejemplo de Dockerfile para el devcontainer:

dockerfile

FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
Ejemplo de devcontainer.json:

json

{
  "name": "Python Dev Container",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "settings": {
    "terminal.integrated.shell.linux": "/bin/bash"
  },
  "postCreateCommand": "pip install -r requirements.txt",
  "extensions": [
    "ms-python.python",
    "ms-toolsai.jupyter"
  ]
}
# requirements.txt ejemplo:

nginx
numpy
pandas
scikit-learn
jupyter



Abre el proyecto en VS Code y haz clic en "Reopen in Container" (puede aparecer automáticamente una sugerencia en la parte inferior derecha).

¡Listo! Ahora tu entorno de desarrollo está corriendo dentro de un contenedor Docker, con todas las dependencias instaladas.