from flask import Flask, redirect, url_for
from src.controller.routes import calcular_hora

app = Flask(
    __name__,
    template_folder='src/view/templates',
    static_folder='src/view/static'
)

app.secret_key = 's3cr3t_k3y_1234567890!@#$%^&*()'

# Registrar los blueprints
app.register_blueprint(calcular_hora)
@app.route('/')
def inicio():
    return redirect(url_for('calculo_hora.calculo_hora_view'))




if __name__ == '__main__':
    app.run(debug=True)