# LlamaIndex Vector_Stores Integration: YDB

Use [YDB](https://ydb.tech) as a vector store for LlamaIndex.

## Installation

```bash
pip install llama-index-vector-stores-ydb
```

## Usage

```python
from llama_index.vector_stores.ydb import YDBVectorStore

vector_store = YDBVectorStore(
    host="localhost",
    port=2136,
    database="/local",
    table_name="my_nodes",
    embed_dim=1536,
)
```

## Fulltext and hybrid search

`text_search` and its alias `sparse` require a `fulltext_relevance` index on
the `text` column. YDB's `FulltextScore` uses the index's term statistics to
compute BM25 scores.

Set `fulltext_index_enabled=True` for BM25 search. Native YDB hybrid search
requires `hybrid_search_enabled=True`, `index_enabled=True`, and
`fulltext_index_enabled=True`. A hybrid query raises `ValueError` if either
index flag is off. The store creates missing indexes on the first query that
needs them and reuses indexes with the configured names. It indexes existing
rows in the same table.

```python
from llama_index.core import VectorStoreIndex

vector_store = YDBVectorStore(
    table_name="my_nodes",
    index_enabled=True,
    fulltext_index_enabled=True,
    hybrid_search_enabled=True,
)
index = VectorStoreIndex.from_vector_store(vector_store)

lexical = index.as_retriever(
    vector_store_query_mode="text_search", similarity_top_k=5
)
hybrid = index.as_retriever(
    vector_store_query_mode="hybrid",
    similarity_top_k=20,
    sparse_top_k=20,
    hybrid_top_k=5,
    alpha=0.5,
)
lexical_results = lexical.retrieve("payment error E104")
hybrid_results = hybrid.retrieve("payment error E104")
```

`text_search` (also available as `sparse`) uses the text query and returns BM25
scores. `sparse_top_k` defaults to `similarity_top_k`.
`hybrid` uses the same text and its embedding. It uses reciprocal rank fusion
with `alpha` as the vector weight and `1 - alpha` as the text weight.
`similarity_top_k` and `sparse_top_k` select the candidate count for each branch.
`hybrid_top_k` defaults to `similarity_top_k` and limits the final result.
Hybrid results are ordered by YDB and have `similarities=None`.
Both modes are available through `query()` and `aquery()`.

The server must support
[fulltext indexes on an `Utf8` primary key](https://ydb.tech/docs/en/dev/fulltext-indexes?version=main)
and, for hybrid queries, [`HybridRank`](https://ydb.tech/docs/en/dev/hybrid-search?version=main).
On YDB 26.3 RC, fulltext indexes on string keys require `EnableFulltextIndexRowId`.
Enable the required server features before using these modes.

`DEFAULT` uses an exact vector scan by default. Set `index_enabled=True` to
use the vector index for `DEFAULT` searches.
Search queries currently reject metadata filters and nonempty `node_ids` or
`doc_ids` restrictions.
