import csv
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin

nlp = spacy.blank('ru')
# the DocBin will store the example documents
db = DocBin()
with open('data/movie_data.csv', encoding='utf8') as inp:
    training_data = csv.DictReader(inp)
    for row in tqdm(training_data, total=50_000):
        doc = nlp.make_doc(row['review'])
        doc.cats = {'pos': int(row['sentiment']), 'neg': 1 - int(row['sentiment'])}
        db.add(doc)
db.to_disk('./data/train.spacy')
