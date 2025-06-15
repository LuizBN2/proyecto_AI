from flask import Blueprint, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np

# Cargar modelo y codificador
modelo = joblib.load('src/model/modelo_entrenado.pkl')
vectorizador = joblib.load('src/model/vector_entrenado.pkl')

calcular_hora = Blueprint('calculo_hora', __name__)

@calcular_hora.route('/calculo_hora', methods=['GET', 'POST'])
def calculo_hora_view():
    if request.method == 'POST':
        try:
            cie10 = request.form.get('cie10').strip().upper()
            sexo_input = request.form.get('sexo')
            edad_input = request.form.get('edad')
            tipo_input = request.form.get('tipo_atencion')

            # Validación
            edad = float(edad_input)
            sexo = 1 if sexo_input == 'M' else 0
            tipo_atencion = int(tipo_input)

            # Codificar y predecir
            cie10_codificado = vectorizador.transform([[cie10]])
            datos_final = np.hstack((cie10_codificado, [[sexo, edad, tipo_atencion]]))
            prediccion = modelo.predict(datos_final)
            horas = prediccion[0]
            minutos = int((horas - int(horas)) * 60)

            resultado = f"{horas:.2f} horas (~{int(horas)} h {minutos} min)"
            return render_template('calculo_hora.html', resultado=resultado)

        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return render_template('calculo_hora.html')

    # Si es GET
    return render_template('calculo_hora.html')


