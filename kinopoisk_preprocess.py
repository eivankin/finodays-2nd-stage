import os
import argparse
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count-for-learning', '-cl', type=int, default=15_000)
    parser.add_argument('--count-for-validation', '-cv', type=int, default=3_000)
    parser.add_argument('--use-neutral', '-n', action='store_true')
    parser.add_argument('--train-size', '-ts', type=float, default=0.7)
    parser.add_argument('--n-process', '-np', type=int, default=4)
    args = parser.parse_args()

    nlp = spacy.load('ru_core_news_lg')
    db_train, db_test, db_validate = DocBin(), DocBin(), DocBin()
    cats = ['neg', 'pos']
    if args.use_neutral:
        cats.append('neu')
    for cat in cats:
        path = Path('data/dataset') / cat
        files = os.listdir(path)[:args.count_for_learning + args.count_for_validation]
        texts = map(lambda f: open(path / f).read(), tqdm(files))
        for i, (doc) in enumerate(nlp.pipe(texts, n_process=args.n_process)):
            doc.cats = {c: c == cat for c in cats}
            if i >= args.count_for_learning:
                db_validate.add(doc)
            if i / args.count_for_learning >= args.train_size:
                db_test.add(doc)
            else:
                db_train.add(doc)
    db_train.to_disk('data/ru_train/train.spacy')
    db_test.to_disk('data/ru_dev/test.spacy')
    db_validate.to_disk('data/ru_validation/validate.spacy')
