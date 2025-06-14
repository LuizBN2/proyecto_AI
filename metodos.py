import joblib

modelo = joblib.load('model/modelo_entrenado.pkl')
vectorizador = joblib.load('model/vector_entrenado.pkl')

class Metodos:
    @staticmethod
    def ingresar_datos(): 
        cie10 = input("Ingresa un código CIE10 para predecir: ")
        nombre_cie10_vectorizado = vectorizador.transform([cie10])
        prediccion = modelo.predict(nombre_cie10_vectorizado)
        dias = prediccion[0]
        horas = int(dias * 24)
        minutos = int((dias * 24 - horas) * 60)
        print(f"La predicción es: {dias:.2f} días (~{horas} horas y {minutos} minutos)")



if __name__ == "__main__":
    Metodos.ingresar_datos()
