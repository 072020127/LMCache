# Official LongBench evaluator

`eval.py` and `metrics.py` are vendored from the upstream LongBench repository
at commit `2e00731f8d0bff23dc4325161044d0ed8af94c1e`. The adapter imports the
same metric functions and translates the nested LMCache cold/hit JSONL format.

Install the optional evaluator dependencies with:

```bash
python -m pip install -r requirements/longbench.txt
```

Run the adapter from the LMCache source root:

```bash
python -m benchmarks.longbench_official.adapter \
  --input /path/to/makv.jsonl \
  --task hotpotqa \
  --dataset-path /path/to/longbench/data/hotpotqa.jsonl \
  --output /path/to/makv.official.summary.json
```

The vendored source hashes are recorded in the repository history and can be
checked with `sha256sum eval.py metrics.py`. The adapter reports both the
internal `[0, 1]` score and the official percentage scale.
