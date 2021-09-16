import spacy
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
nlp = spacy.load('trained-ru-20k-70p/model-best')
EPS = 0.15
VERDICT_TO_MESSAGE_TYPE = {'positive': 'success', 'negative': 'danger', 'neutral': 'secondary'}


def get_verdict(text: str):
    doc = nlp(text)
    pos, neg = doc.cats['pos'], doc.cats['neg']
    return {'verdict': (
        'neutral' if abs(pos - neg) < EPS else 'positive' if pos > neg else 'negative'),
        'cats': doc.cats}


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    return jsonify(get_verdict(request.json['text']))


@app.route('/', methods=['GET', 'POST'])
def analyze():
    verdict, v_type = None, None
    if request.method == 'POST':
        verdict = get_verdict(request.form['text'])
        v_type = VERDICT_TO_MESSAGE_TYPE[verdict['verdict']]
    return render_template('analyze.html', cats=verdict, text=request.form['text'], type=v_type)


if __name__ == '__main__':
    app.run('0.0.0.0')
