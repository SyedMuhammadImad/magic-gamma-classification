# MAGIC gamma classification notebook

Completed academic model comparison. Repaired notebook and command-line workflow use seeded splits and training-only preprocessing. Numeric results are in `metrics.json`; checks and limits are in `VERIFICATION.json`.

Install `requirements.txt`, then run `python experiment.py --data PATH_TO_ORIGINAL_DATA --output metrics.json`. Original source datasets remain local; model weights, pictures and videos are excluded.

The notebook keeps the setup → experiment → results pattern of the other published labs.
