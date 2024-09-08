import logging
import fire
import sqlalchemy
import json
import app
import tqdm
import dotenv

import difflib

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

def diff_str(str1, str2):
    """Prints a minimal diff between two strings."""
    differ = difflib.Differ()
    diff = differ.compare(str1.splitlines(), str2.splitlines())
    return diff


class AppDemo:
    def rewrite_text(self, sentence):
        response = app._annotate_terms(sentence)
        return response

    def generate_article(self, article):
        assert article
        config = app.GenerateConfig(article=article)
        content = []
        progress = tqdm.tqdm(total=1000)
        for resp in app._generate_article(config):
            if "choices" not in resp:
                continue
            choices = resp["choices"][0]
            if "delta" not in choices:
                continue
            if "content" not in choices["delta"]:
                continue
            delta = resp["choices"][0]["delta"]["content"]
            content.append(delta)
            progress.update(len(delta.split()))
        print("".join(content))

    def annotate_article(self, article):
        assert article
        config = app.GenerateConfig(article=article)
        content = []
        progress = tqdm.tqdm(total=1000)
        for resp in app._generate_article(config):
            choices = resp["choices"][0]
            if "content" not in choices["delta"]:
                continue
            delta = resp["choices"][0]["delta"]["content"]
            content.append(delta)
            progress.update(len(delta.split()))

        annotated = app._annotate_text(config, "".join(content))

        stream_annotation = []
        for delta in app._batch_response(app._generate_article(config)):
            stream_annotation.append(app._annotate_text(config, delta))

        stream_annotation = "".join(stream_annotation)
        print(len(annotated), len(stream_annotation))
        diff = list(diff_str(annotated, stream_annotation))
        assert diff == [], "\n".join(diff)

    def lookup_article_config(self, article):
        return app._lookup_article_config(article)

    def clear_cache(self):
        with app.db.begin() as conn:
            conn.execute(sqlalchemy.delete(app.db.cache))


if __name__ == "__main__":
    dotenv.load_dotenv()
    fire.Fire(AppDemo)
