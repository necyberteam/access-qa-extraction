# TRACE-TOUR.extract: CLI extraction pipeline

> Trace of: `qa-extract extract compute-resources allocations --max-entities 2 --incremental`

## Overview

```
extract()
├── Config.from_env()                              # read env vars, build server configs
├── Apply CLI overrides (--max-entities, etc.)
├── IncrementalCache(output_dir)                   # load .extraction_cache.json
│
├── FOR EACH SERVER:
│   ├── Extractor.__init__()                       # create LLM + judge clients
│   ├── Fetch entities                             # MCP call or direct API
│   ├── Build battery system prompt                # once per domain, reused
│   │
│   └── FOR EACH ENTITY:
│       ├── Clean data → hash → cache check
│       │
│       ├── [CACHE HIT] → replay stored pairs
│       ├── [CACHE MISS] →
│       │   ├── LLM call 1: battery                # one Q&A per field group
│       │   ├── LLM call 2: discovery              # find what battery missed
│       │   ├── LLM call 3: judge                  # score each pair (cheaper model)
│       │   └── cache.store(hash, pairs)
│       │
│       └── Collect raw_data for comparisons
│
├── cache.save()                                   # persist to disk
├── ComparisonGenerator.generate()                 # cross-entity Q&As, no LLM
├── Print summary table
├── JSONLWriter.write_all()                        # one .jsonl per domain + comparisons
└── [optional] ArgillaClient.push_pairs()          # entity-replace to human review
```

## Trace

```
extract[1]   cli.py:77  extract()
extract[2]   ├── config.py  Config.from_env()
extract[3]   ├── cli.py:136  Apply CLI overrides
extract[4]   ├── incremental.py:34  IncrementalCache(output_dir)
             │
extract[5]   ├── asyncio.run(run_all())
             │   │
extract[6]   │   ├── cli.py:45  run_extraction("compute-resources", ...)
             │   │   ├── ComputeResourcesExtractor.__init__()
             │   │   │   └── llm_client.py:258  get_llm_client()
             │   │   │
extract[7]   │   │   └── base.py:68  BaseExtractor.run()
             │   │       ├── mcp_client.py:19  MCPClient.__aenter__()
             │   │       │
extract[8]   │   │       └── compute_resources.py:71  extract()
extract[9]   │   │           ├── mcp_client.py:27  call_tool("search_resources", {})
extract[10]  │   │           ├── question_categories.py:298  build_battery_system_prompt()
             │   │           │
extract[11]  │   │           └── FOR EACH ENTITY:
             │   │               ├── mcp_client.py:27  call_tool("get_resource_hardware", {id})
             │   │               ├── compute_resources.py:210  _clean_resource_data()
             │   │               ├── compute_resources.py:233  _clean_hardware_data()
             │   │               ├── incremental.py:16  compute_entity_hash()
             │   │               ├── incremental.py:48  cache.is_unchanged()
             │   │               │
extract[12]  │   │               ├── [CACHE HIT] → replay stored pairs
extract[13]  │   │               ├── [CACHE MISS] →
             │   │               │   ├── compute_resources.py:257  _generate_qa_pairs()
             │   │               │   │   ├── question_categories.py:340  build_user_prompt()
extract[14]  │   │               │   │   ├── llm_client.py  llm.generate() (battery)
             │   │               │   │   ├── question_categories.py:314  build_discovery_system_prompt()
extract[15]  │   │               │   │   ├── llm_client.py  llm.generate() (discovery)
             │   │               │   │   └── models.py:51  QAPair.create() x N
             │   │               │   │
extract[16]  │   │               │   ├── judge.py:72  evaluate_pairs()
             │   │               │   │   └── llm_client.py  judge_client.generate()
             │   │               │   │
extract[17]  │   │               │   └── incremental.py:68  cache.store(hash, pairs)
             │   │               │
             │   │               └── Collect raw_data for comparisons
             │   │
             │   └── cli.py:45  run_extraction("allocations", ...)
             │       └── allocations.py:54  run() (overrides BaseExtractor)
             │           └── allocations.py:128  extract()
             │               ├── allocations.py:86  _fetch_all_projects()
             │               └── (same per-entity loop as extract[11]-[17])
             │
extract[18]  ├── incremental.py:78  cache.save()
             │
extract[19]  ├── comparisons.py:55  ComparisonGenerator.generate()
             │
extract[20]  ├── cli.py:213  Print Rich summary table
             │
extract[21]  ├── jsonl_writer.py:45  write_all(results)
             │   ├── write("compute-resources", pairs)
             │   ├── write("allocations", pairs)
             │   └── write("comparisons", pairs)
             │
extract[22]  └── [optional] cli.py:333  _push_pairs_to_argilla()
```

## Files

| Order | File | Role |
|---|---|---|
| 1 | `cli.py` | Parse CLI args, dispatch extractors, collect results, write output |
| 2 | `config.py` | Read env vars into server and extraction config dataclasses |
| 3 | `incremental.py` | Hash entities, compare hashes, load/save cache |
| 4 | `llm_client.py` | Factory for LLM backends (Anthropic, OpenAI, local, transformers) |
| 5 | `base.py` | Open MCP connection, call abstract `extract()` |
| 6 | `mcp_client.py` | POST to MCP server, unwrap response envelope |
| 7 | `compute_resources.py` | Fetch + clean HPC resources, per-entity LLM + judge loop |
| 8 | `allocations.py` | Override `run()`, paginate public API, same per-entity loop |
| 9 | `question_categories.py` | Build battery + discovery prompts, define field guidance |
| 10 | `judge.py` | Score all pairs per entity (faithfulness, relevance, completeness) |
| 11 | `models.py` | `QAPair.create()` — canonical data model |
| 12 | `comparisons.py` | Group entities by shared attributes, create cross-entity pairs |
| 13 | `jsonl_writer.py` | Write `QAPair.model_dump_json()` lines to `.jsonl` files |
