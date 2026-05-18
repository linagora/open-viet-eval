# OpenRAG setup notes for SCORE

This file summarizes the changes and the exact steps needed to recreate the working setup after cloning.

## Summary of code/config changes

1) Split embeddings config for OpenAI-compatible providers
- Added separate embeddings base URL and API key for the OpenAI provider in [score/settings.py](../score/settings.py).
- Updated the OpenAI client initialization so embeddings can use a different base URL and/or key in [llm/client.py](../llm/client.py).
- Documented new env vars in [.env.example](../.env.example).

New env vars:
- OPENAI_BASE_URL
- OPENAI_EMBEDDING_BASE_URL
- OPENAI_EMBEDDING_API_KEY

2) Docker build dependencies for pycairo
- Added pkg-config and cairo dev libs to the builder image.
- Added cairo runtime libs to the runtime image.
- Changes are in [Dockerfile](../Dockerfile).

3) Embedding dimension alignment
- Set embedding dimensions to 1024 for Qwen3-Embedding-0.6B in [config.yaml](../config.yaml).

## Required runtime configuration

### .env (OpenRAG as OpenAI-compatible provider)

Use OpenRAG for both chat and embeddings (since /v1/embeddings works):

```
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://chat.lucie.ovh.linagora.com/v1/
OPENAI_API_KEY=sk-<openrag-token>
```

Note: Model IDs are read from config.yaml, not from .env.

### config.yaml (model IDs and dimensions)

```
llm:
  provider: openai
  chat_model: Mistral-Small-3.2-24B-Instruct-2506-FP8
  embedding_model: Qwen3-Embedding-0.6B
  embedding_dimensions: 1024
```

## Clean setup steps (from a fresh clone)

1) Create and activate a venv

```
python3.12 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Extra dev/test dependencies used locally

These are used by tests in this repo:

```
pip install pytest pytest-django sentence-transformers
```

4) Install spaCy model(s) needed by tests

```
python -m spacy download en_core_web_sm
```

5) Configure .env

Copy and update .env:

```
cp .env.example .env
```

Then set the OpenRAG values in .env (see the section above).

6) Apply migrations and collect static

```
python manage.py migrate --run-syncdb
python manage.py collectstatic --noinput
```

7) Reset vector store if embedding dimensions change

If the embedding dimensions were changed or the vec store was created with a different size, remove the sqlite-vec file:

```
rm -f data/vec.sqlite3
```

8) Start Redis (if using the default broker)

```
redis-server
```

If Redis is not available, use the database broker in .env:

```
CELERY_BROKER_BACKEND=database
```

Then start the worker with:

```
celery -A score worker -l info -P solo
```

9) Start the app

```
python manage.py runserver
```

10) Run tests

```
pytest tests/
```

## Common issues and fixes

1) Redis connection refused
- Make sure Redis is running, or switch to database broker with CELERY_BROKER_BACKEND=database.

2) Embedding dimension mismatch
- Ensure config.yaml has embedding_dimensions=1024 for Qwen3-Embedding-0.6B and delete data/vec.sqlite3 before re-ingesting.

3) Hugging Face cache permission error
- Use a local cache directory:

```
export HF_HOME=/home/<user>/Desktop/SCORE/.hf-cache
export TRANSFORMERS_CACHE=/home/<user>/Desktop/SCORE/.hf-cache/transformers
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
```

4) spaCy model missing
- Install the model (see step 4).

5) OpenAI 401 from embeddings
- Ensure OPENAI_API_KEY is a valid token for OpenRAG and /v1/embeddings works.

## Validation

Quick embedding sanity check against OpenRAG:

```
curl -X POST 'https://chat.lucie.ovh.linagora.com/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-<openrag-token>' \
  -d '{"model":"Qwen3-Embedding-0.6B","input":"hello"}'
```

## Reset
pkill -f "manage.py runserver"
pkill -f "celery -A score worker"
sudo pkill -f "redis-server"

rm -f data/db.sqlite3
rm -f data/vec.sqlite3
rm -f data/celery_broker.sqlite3
rm -rf media/*
rm -rf staticfiles/*

find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete

source .venv/bin/activate
python manage.py migrate --run-syncdb
python manage.py collectstatic --noinput
celery -A score worker -l info
python manage.py runserver

python scripts/load_sample_data.py