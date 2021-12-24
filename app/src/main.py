from flask import Flask


app = Flask(__name__)


if __name__ == '__main__':

    port = 7000
    app.run(host='0.0.0.0', port=port, debug=True)
