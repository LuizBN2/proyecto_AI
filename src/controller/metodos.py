import joblib
import numpy as np

# Cargar modelo y vectorizador
modelo = joblib.load('C:/Users/apena/proyecto_AI/src/model/modelo_entrenado.pkl')
vectorizador = joblib.load('C:/Users/apena/proyecto_AI/src/model/vector_entrenado.pkl')


class Metodos:
    @staticmethod
    def ingresar_datos():
        print("=== Predicción de duración en horas ===")

        cie10 = input("Código CIE10: ").strip().upper()
        sexo_input = input("Sexo (M:1/F:2): ").strip().upper()
        edad_input = input("Edad: ").strip()
        tipo_input = input("Tipo de atención (1=Consulta / 2=Urgencias / 3=Otro): ").strip()

        try:
            edad = float(edad_input)
            sexo = 1 if sexo_input == 'M' else 0
            tipo_atencion = int(tipo_input)
        except ValueError:
            print("Entrada inválida. Asegúrate de que los valores sean correctos.")
            return

        try:
            # Codificar solo CIE10
            cie10_codificado = vectorizador.transform([[cie10]])

            # Unir con los valores numéricos
            datos_final = np.hstack((cie10_codificado, [[sexo, edad, tipo_atencion]]))

            # Predicción
            prediccion = modelo.predict(datos_final)
            horas = prediccion[0]
            minutos = int((horas - int(horas)) * 60)

            print(f"\nPredicción: {horas:.2f} horas (~{int(horas)} h {minutos} min)")

        except Exception as e:
            print(f"\nError al hacer la predicción: {e}")
            print("Verifica que los datos estén bien codificados y que coincidan con el modelo entrenado.")

if __name__ == "__main__":
    Metodos.ingresar_datos()
