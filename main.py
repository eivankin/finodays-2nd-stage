import spacy
from flask import Flask, jsonify, request

app = Flask(__name__)
nlp = spacy.load('trained-ru-20k-70p/model-best')
EPS = 0.15


@app.route('/analyze', methods=['POST'])
def analyze():
    doc = nlp(request.json['text'])
    pos, neg = doc.cats['pos'], doc.cats['neg']
    return jsonify({'verdict': (
        'neutral' if abs(pos - neg) < EPS else 'positive' if pos > neg else 'negative'),
        'cats': doc.cats})


if __name__ == '__main__':
    app.run('0.0.0.0')
