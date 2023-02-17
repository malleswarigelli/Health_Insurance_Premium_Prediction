
from flask import Flask, request, jsonify 


app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return "start ML project"


if __name__ == "__main___":
    app.run()