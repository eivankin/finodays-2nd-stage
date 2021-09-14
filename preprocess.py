import os
import csv
import argparse
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split


def save_data(path: Path, data: list) -> None:
    print('Converting...')
    for row in tqdm(data):
        doc = nlp.make_doc(row['review'])
        doc.cats = {'pos': int(row['sentiment']), 'neg': 1 - int(row['sentiment'])}
        db.add(doc)
    if not path.parent.exists():
        os.mkdir(path.parent)
    print(f'Saving to "{path}"...')
    db.to_disk(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', default='en', choices=['en', 'ru'])
    parser.add_argument('--input', '-i', default='data/movie_data.csv', type=Path)
    parser.add_argument('--train-output', '-to', default='data/train.spacy', type=Path)
    parser.add_argument('--dev-output', '-do', default='data/dev/test.spacy', type=Path)
    parser.add_argument('--test-size', '-ts', type=float, default=0.5)
    args = parser.parse_args()

    nlp = spacy.blank(args.language)
    db = DocBin()
    with open(args.input, encoding='utf8') as inp:
        data = list(csv.DictReader(inp))
        train, test = train_test_split(data, test_size=args.test_size)
        save_data(args.train_output, train)
        save_data(args.dev_output, test)
