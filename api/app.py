import functools
import coloredlogs
import hashlib
import json
import os
import re
import typing
import logging
import fastapi

import httpx
import pydantic
import sqlalchemy.dialects.postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert
import sqlalchemy
import sqlalchemy.exc
from openai import OpenAI
from pydantic import BaseModel

import dotenv
import starlette

dotenv.load_dotenv(".env.production")


coloredlogs.install(level="INFO")

logger = logging.getLogger(__name__)
app = fastapi.FastAPI()


class DBConfig(pydantic.BaseModel):
    db_type: str
    db_name: str
    db_user: typing.Optional[str] = None
    db_pass: typing.Optional[str] = None
    db_host: typing.Optional[str] = None
    db_port: typing.Optional[str] = None

    # Initialize from environment
    @classmethod
    def from_env(cls):
        return cls(
            db_user=os.environ.get("DB_USER"),
            db_pass=os.environ.get("DB_PASS"),
            db_type=os.environ.get("DB_TYPE"),
            db_name=os.environ.get("DB_NAME"),
            db_host=os.environ.get("DB_HOST"),
            db_port=os.environ.get("DB_PORT"),
        )


class DB:
    meta: sqlalchemy.MetaData
    cache: sqlalchemy.Table
    article: sqlalchemy.Table
    _engine = None

    def __init__(self, config: DBConfig):
        self.meta = sqlalchemy.MetaData()
        self.cache = sqlalchemy.Table(
            "cache",
            self.meta,
            sqlalchemy.Column("key", sqlalchemy.String, primary_key=True),
            sqlalchemy.Column("value", sqlalchemy.JSON),
        )
        self.article = sqlalchemy.Table(
            "article",
            self.meta,
            sqlalchemy.Column("article", sqlalchemy.String, primary_key=True),
            sqlalchemy.Column("prompt", sqlalchemy.String),
            sqlalchemy.Column("included_topics", sqlalchemy.String),
            sqlalchemy.Column("excluded_topics", sqlalchemy.String),
        )

        self._create_engine(config)
        # self.meta.drop_all(self._engine)
        self.meta.create_all(self._engine)

    def _create_engine(self, config: DBConfig):
        if config.db_type == "sqlite":
            self._engine = sqlalchemy.create_engine(f"sqlite:///{config.db_name}")
        elif config.db_type == "postgres":
            self._engine = sqlalchemy.create_engine(
                f"postgresql://{config.db_user}:{config.db_pass}@{config.db_host}:{config.db_port}/{config.db_name}"
            )
        else:
            raise ValueError("Unsupported database type")

    def begin(self):
        return self._engine.begin()


db = DB(config=DBConfig.from_env())


class GenerateConfig(BaseModel):
    article: str
    prompt: str = ""
    model: str = "gpt-3.5-turbo"
    annotate_model: str = "gpt-4o-mini"
    included_topics: typing.List[str] = []
    excluded_topics: typing.List[str] = []


def _lookup_generate_config(article):
    article = article.replace("_", " ")
    return GenerateConfig(
        article=article,
        included_topics=[],
        excluded_topics=[],
    )


class DBCache:
    _cache: dict
    _db: DB

    def __init__(self, db: DB) -> None:
        self._cache = {}
        self._db = db

    def fetch(self, key):
        if key in self._cache:
            return self._cache[key]

        command = sqlalchemy.select(db.cache.c.key, db.cache.c.value).where(
            db.cache.c.key == key
        )
        with db.begin() as conn:
            cached = conn.execute(command).fetchone()
            logging.info("Cache for %s -> %s", key, cached)
            if cached:
                self._cache[key] = cached[1]
                return cached[1]
            else:
                return None

    def put(self, key, value: typing.Any):
        command = (
            pg_insert(self._db.cache)
            .values(
                key=key,
                value=sqlalchemy.cast(value, sqlalchemy.JSON),
            )
            .on_conflict_do_update(
                index_elements=["key"],
                set_=dict(value=value),
            )
        )
        with self._db.begin() as conn:
            self._cache[key] = value
            conn.execute(command)
            logging.info("Added %s to cache.", key)


_cache = DBCache(db)


def memoize(streaming=False):
    def _cache_key(prefix, *args, **kwargs):
        # handle instance methods
        if isinstance(args[0], pydantic.BaseModel):
            key = args[0]
        else:
            key = args[1]

        return (
            prefix
            + "::"
            + hashlib.sha256(key.model_dump_json().encode("utf-8")).hexdigest()
        )

    if streaming:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = _cache_key(func.__name__, *args, **kwargs)
                cached_result = _cache.fetch(key)
                if cached_result:
                    logging.info("Cache hit for %s", key)
                    for chunk in cached_result:
                        yield chunk
                else:
                    logging.info("Cache miss for %s", key)
                    cache_result = []
                    for chunk in func(*args, **kwargs):
                        cache_result.append(chunk)
                        yield chunk
                    logging.info("Updating cache: %s -> %s", key, len(cache_result))
                    _cache.put(key, cache_result)

            return wrapper

    else:

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = _cache_key(func.__name__, *args, **kwargs)
                cached_result = _cache.fetch(key)
                if cached_result:
                    logging.info("Cache hit for %s", key)
                    return cached_result
                else:
                    logging.info("Cache miss for %s", key)
                    result = func(*args, **kwargs)
                    _cache.put(key, result)
                    logging.info("Updating cache: %s -> %s", key, type(result))
                    return result

            return wrapper

    return decorator


class LLMRequest(pydantic.BaseModel):
    model: str
    seed: int = 42
    messages: typing.List[typing.Dict]
    response_format: typing.Optional[typing.Dict] = None


class OpenAIClient:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    @memoize(streaming=True)
    def fetch_streaming(self, request: LLMRequest):
        """Fetch `request` using the OpenAI client.

        Use a cache keyed on the request JSON to avoid re-fetching. Streaming requests
        are cached as a sequence of individual responses to ensure the same behavior
        on cache hit and cache miss."""
        kwarg = request.model_dump()
        kwarg["stream"] = True

        full_response = []
        response = self.client.chat.completions.create(**kwarg)
        for chunk in response:
            chunk = chunk.model_dump(exclude_unset=True)
            full_response.append(chunk)
            yield chunk

    @memoize(streaming=False)
    def fetch(self, request: LLMRequest):
        kwarg = request.model_dump()
        kwarg["stream"] = False
        response = self.client.chat.completions.create(**kwarg)
        response = response.model_dump(exclude_unset=True)
        return response


openai_client = OpenAIClient()


generate_prompt = """
Adhere to the following guidelines:

* Target an expert, not a casual reader.
* Highlight recent research about the topic.
* Be complete: target 1000 words or more.
* Don't be verbose or chatty, target someone in a hurry.
* Use markdown format.
"""

test_prompt = """
Do you know anything about the following topic?
Answer truthfully. Do not make anything up.
Use JSON format.

Example response.

Topic: "ABCQ Therapy"
{ "known": false, "reason": "I have never heard of ABCQ Therapy." }

Topic: "MHC-I"
{ "known": true, "reason": "MHC-1 is a gene." }

Topic: "MHC-II"
{ "known": true, "reason": "MHC-2 is a gene." }
"""

annotate_prompt = """
Identify any proper nouns or terms that may be unfamiliar to the reader.
For each term, map the form in the sentence to a normalized form of the term suitable for an article title.
Emit a JSON response.

Example: "MHC-I, or Major Histocompatibility Complex class I, is a group of cell surface proteins found on all nucleated cells in the body."
Output: { "terms": {
    "MHC-I": "Major Histocompatibility Complex class I",
    "nucleated cells": "Nucleated Cell",
    "proteins": "Protein"
} }

Example: "A pear is a fruit produced by the Pyrus genus of trees in the Rosaceae family."
Output: { "terms": { "Pyrus": "Pyrus (Genus)", "Rosaceae": "Rosaceae (Family)" } }
"""


def _generate_article(generate_config: GenerateConfig):
    """Generate an article using the OpenAI API."""
    # check if the model knows about the topic first
    messages = [
        {"role": "system", "content": test_prompt},
        {"role": "user", "content": f"Topic: {generate_config.article}"},
    ]
    response = openai_client.fetch(
        LLMRequest(model=generate_config.model, messages=messages)
    )
    logger.debug("Topic is known: %s", response)
    test_response = response["choices"][0]["message"]["content"]
    try:
        test_response = json.loads(test_response)
        if not test_response.get("known", False):
            return [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "I do not know anything about this topic."
                            }
                        }
                    ]
                }
            ]
    except json.decoder.JSONDecodeError:
        logger.error("Error parsing JSON response from OpenAI API: %s", test_response)
        return [
            {
                "choices": [
                    {"delta": {"content": "I do not know anything about this topic."}}
                ]
            }
        ]

    default_prompt = (
        f'Describe "{generate_config.article}" to a college-educated reader.'
    )
    prompt = generate_config.prompt or default_prompt

    messages = [
        {"role": "system", "content": generate_prompt},
        {"role": "user", "content": prompt},
    ]
    return openai_client.fetch_streaming(
        LLMRequest(model=generate_config.model, messages=messages)
    )


def _find_annotations(generate_config: GenerateConfig, article_text):
    """Identify any terms in `article_text` which should have annotations."""
    annotate_requests = [
        {
            "role": "system",
            "content": annotate_prompt,
        },
        {
            "role": "system",
            "content": f'Do not provide a definition for "{generate_config.article}."',
        },
        {
            "role": "user",
            "content": article_text,
        },
    ]

    response = openai_client.fetch(
        LLMRequest(
            model=generate_config.annotate_model,
            messages=annotate_requests,
            response_format={"type": "json_object"},
        )
    )
    content = response["choices"][0]["message"]["content"]
    try:
        annotation_dict = json.loads(content)
        return annotation_dict
    except json.decoder.JSONDecodeError:
        return {"terms": {}}


def _annotate_text(generate_config: GenerateConfig, md_text: typing.AnyStr):
    """Annotate sentences with unfamiliar terms."""
    annotation_dict = _find_annotations(generate_config, md_text)

    # for each term in annotation_dict['terms'], replace instances of the term
    # with a markdown link to the corresponding article
    # of the form [term](/view/{term})
    for term, article_title in annotation_dict.get("terms", {}).items():
        # Don't annotate the current article
        if term.lower().replace(" ", "_") == generate_config.article.lower().replace(
            " ", "_"
        ):
            continue

        article_title = article_title.replace(" ", "_")

        term = re.escape(term)
        # replace all non-word characters in term with underscores
        md_text = re.sub(
            rf"(?i)\b{term}\b",
            lambda m: f"[{m.group(0)}](/view/{article_title})",
            md_text,
        )
    return md_text


def _batch_response(generator, batch_size=100, max_content_size=256):
    """Batch responses from OpenAI generator to reduce round-trips."""
    batch = []
    for chunk in generator:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            content = delta["content"]
            batch.append(content)

        batch_str = "".join(batch)
        content_size = len(batch_str)

        # Yield batches immediately if we find a paragraph boundary.
        split = re.search(r"\n", batch_str)
        if split:
            yield batch_str[: split.end()]
            batch = [batch_str[split.end() :]]
            content_size = len(batch[0])
        else:
            if len(batch) >= batch_size or content_size >= max_content_size:
                batch_str = "".join(batch)

    if batch:
        yield "".join(batch)


@memoize(streaming=True)
def _stream_article(generate_config: GenerateConfig):
    for content in _batch_response(_generate_article(generate_config)):
        content = _annotate_text(generate_config, content)
        yield json.dumps({"content": content}) + "\n"

@app.get("/api/article/view")
def view_article_by_name(article: str, model: str = "gpt-3.5-turbo"):
    generate_config = _lookup_generate_config(article)
    generate_config.model = model
    logger.info(f'Fetching article "{article}"')
    return fastapi.responses.StreamingResponse(
        _stream_article(generate_config),
        media_type="application/json",
    )


# let's not crawl an infinite site..
@app.get("/robots.txt", response_class=fastapi.responses.PlainTextResponse)
def robots_txt():
    robots_content = """
    User-agent: *
    Disallow: /
    Allow: /$
    Allow: /about$
    """
    return robots_content


@app.get("/{path:path}")
async def proxy_to_frontend(request: fastapi.Request, path: str):
    client = httpx.AsyncClient(base_url="http://localhost:3001")
    url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
    headers = {"Upgrade": "none", **request.headers}

    rp_req = client.build_request(
        request.method, url, headers=headers, content=await request.body()
    )
    rp_resp = await client.send(rp_req, stream=True)
    return fastapi.responses.StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=starlette.background.BackgroundTask(rp_resp.aclose),
    )
