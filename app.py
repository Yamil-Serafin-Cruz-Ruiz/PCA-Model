from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Cargar modelos
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("pca_model.pkl", "rb") as f:
    pca = joblib.load(f)

with open("survived_knn_classifier.pkl", "rb") as f:
    knn = joblib.load(f)

# Codificación de cabina basada en letra
def codificar_cabina(cabina_str):
    if not cabina_str:
        return 0
    letra = cabina_str[0].upper()
    letras_cabina = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    return letras_cabina.get(letra, 0)

@app.route('/')
def index():
    return render_template("formulario.html")

@app.route('/predecir', methods=["POST"])
def predecir():
    try:
        # Recoger datos del formulario
        sex_male = int(request.form.get("Sex_male"))
        age = float(request.form.get("Age"))
        fare = float(request.form.get("Fare"))
        pclass = int(request.form.get("Pclass"))
        cabina_str = request.form.get("Cabina")
        cabina_codificada = codificar_cabina(cabina_str)

        # Asignar valores por defecto para columnas faltantes
        sibsp = 0
        parch = 0
        embarked = 0

        # Construir el vector de entrada (8 columnas esperadas)
        # ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Sex_male', 'Cabina']
        features = np.array([[pclass, age, sibsp, parch, fare, embarked, sex_male, cabina_codificada]])

        # Preprocesamiento
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Predicción
        prediction = knn.predict(features_pca)
        resultado = "Sobrevive" if prediction[0] == 1 else "No sobrevive"

        return render_template("formulario.html", resultado=resultado)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
