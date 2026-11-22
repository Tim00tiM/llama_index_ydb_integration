# Testing Setup

If a container named `test-ydb` already exists, remove it before starting a
fresh test database.

```bash
docker rm -f test-ydb
```

Start a fresh local YDB for integration tests. With
`YDB_USE_IN_MEMORY_PDISKS=true`, stopping YDB clears its storage in RAM.
Create a fresh container for the next run.

```bash
docker run -d --rm --name test-ydb -p 2136:2136 \
  -e YDB_USE_IN_MEMORY_PDISKS=true -h localhost \
  ydbplatform/local-ydb:trunk
```

The fulltext and hybrid integration tests need server support for fulltext
indexes on `Utf8` primary keys and for `HybridRank`. Enable these features if
the server disables them by default.

Wait for `docker inspect --format '{{.State.Health.Status}}' test-ydb` to report
`healthy`. Then run all tests from the `llama-index-vector-stores-ydb` package
directory.

```bash
uv run pytest tests
```

Use `docker stop test-ydb` to stop YDB. Docker removes the container
automatically because it was started with `--rm`.
