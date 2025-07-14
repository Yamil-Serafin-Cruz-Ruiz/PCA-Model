from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelos
with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("pca_model.pkl", "rb") as f:
    pca = joblib.load(f)

with open("survived_knn_classifier.pkl", "rb") as f:
    knn = joblib.load(f)

# Función para codificar la letra de la cabina
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
        # Obtener los datos enviados desde el formulario
        sex_male = int(request.form.get("Sex_male"))
        age = float(request.form.get("Age"))
        fare = float(request.form.get("Fare"))
        pclass = int(request.form.get("Pclass"))
        cabina_str = request.form.get("Cabina")
        cabina_codificada = codificar_cabina(cabina_str)

        # Asignar valores por defecto a las columnas que no se capturan en el formulario
        sibsp = 0
        parch = 0
        embarked = 0

        # Construir el arreglo con las 8 columnas requeridas por el scaler
        # Orden: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Sex_male', 'Cabina']
        features = np.array([[pclass, age, sibsp, parch, fare, embarked, sex_male, cabina_codificada]])

        # Escalar las 8 columnas
        features_scaled = scaler.transform(features)

        # Seleccionar las columnas esperadas por PCA en el orden correcto:
        # ['Sex_male', 'Age', 'Fare', 'Pclass', 'Cabina'] → [6, 1, 4, 0, 7]
        features_para_pca = features_scaled[:, [6, 1, 4, 0, 7]]

        # Aplicar PCA
        features_pca = pca.transform(features_para_pca)

        # Hacer predicción
        prediction = knn.predict(features_pca)
        resultado = "Sobrevive" if prediction[0] == 1 else "No sobrevive"

        # Mostrar el resultado en el mismo formulario
        return render_template("formulario.html", resultado=resultado)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
