from flask import Flask, request, jsonify

app = Flask('ping')

@app.route('/', methods=['GET'])