import os
from flask import Flask
from controllers.house_prediction_controller import house_prediction_api

app = Flask(__name__)

# register API house predictions controller
app.register_blueprint(house_prediction_api)

if __name__ == '__main__':
    # for heroku to get the assigned port
    port = int(os.environ.get('PORT', 7000))
    app.run(host='0.0.0.0', port=port, debug=True)
