# Pollpear_ai

The recommendation service. It reads the poll and interaction history from PostgreSQL,
trains a ranking model, and serves a personalized feed over HTTP — kept off the request
path so a slow model never blocks a vote.

Part of **[Pollpear](https://github.com/transummer2026/Pollpear)**, a real-time collaborative polling platform.

## Stack

Python 3.14 · FastAPI + uvicorn · scikit-learn, NumPy, pandas, joblib · psycopg +
SQLAlchemy to read the database · httpx · matplotlib for evaluation plots ·
`prometheus-fastapi-instrumentator` to expose metrics.

## Layout

| Path | Role |
| :--- | :--- |
| `src/data/loader.py` | Pulls the training set out of PostgreSQL |
| `src/training/trainer.py` | Fits the model |
| `src/models/` | The feed model and artifact persistence |
| `src/scoring/recommender.py` | Ranks candidate polls for a user |
| `src/evaluation/evaluate.py` | Offline metrics |
| `src/api/main.py` | The FastAPI app |
| `src/vault/` | Reads its database credentials from HashiCorp Vault |
| `notebooks/` | Exploration — not part of the service |

## Running it

Through the stack, which gives it a seeded database and Vault:

```sh
git clone --recursive https://github.com/transummer2026/Pollpear.git
cd Pollpear && make up_prod && make seed_prod
```

Standalone: `make install`, then `make test`, `make lint`, `make coverage`,
`make audit`. `make ci` chains lint, tests, coverage and build.

Full setup instructions live in the [superproject README](https://github.com/transummer2026/Pollpear#readme).
