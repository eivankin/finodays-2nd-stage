from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def analyze():
    print(request.json)
    return jsonify({'message': 'text', 'result': {'positive': .0, 'negative': .0, 'neutral': .0}})


if __name__ == '__main__':
    app.run('0.0.0.0')
