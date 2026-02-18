# Guided Tour: Following a Q&A Pair From Birth to Disk

> **Note (2026-02-18):** This document reflects the **old 4-pass pipeline** (categories + factoid + bonus + judge). The codebase has since moved to a **3-pass freeform pipeline** (freeform LLM + factoid + judge) on branch `spike/freeform-extraction`. The overall flow is similar — the main difference is that Passes 1 and 3 (categories + bonus) were merged into a single freeform LLM call that produces variable pair counts. See `docs/design-extraction-rethink-2026-02-18.md` for the design rationale and results. This walkthrough still accurately describes the factoid, judge, cache, comparison, and output stages.

A chronological trace of the code path, from when you type `qa-extract extract`
to when a JSONL line hits the filesystem. No metaphors, no themes — just the
code in the order it runs, with enough narration to make it stick.

We'll trace one concrete invocation:

```bash
qa-extract extract compute-resources allocations --max-entities 2 --incremental
```

---

## Step 1: The CLI Parses Your Command

**File:** `src/access_qa_extraction/cli.py:76`
**Function:** `extract()`

Typer decorates this function as `@app.command()`. When you type
`qa-extract extract compute-resources allocations`, Typer passes
`servers=["compute-resources", "allocations"]` as a list.

The function immediately does three things:

### 1a. Build the config

```python
config = Config.from_env()   # cli.py:129
```

This reads env vars to populate `MCPServerConfig` objects (one per domain)
and `ExtractionConfig` objects (one per domain). The config knows things like:
- `compute-resources` lives at `http://localhost:3002`
- `allocations` lives at `http://localhost:3006`
- `max_tokens = 2048` (default)

### 1b. Apply CLI flag overrides

```python
# cli.py:131-139
if search_limit is not None or max_queries is not None or max_entities is not None:
    for name in config.extraction:
        if search_limit is not None:
            config.extraction[name].search_limit = search_limit
        if max_queries is not None:
            config.extraction[name].max_queries = max_queries
        if max_entities is not None:
            config.extraction[name].max_entities = max_entities
```

This mutates every `ExtractionConfig` in the dict. For our invocation, only
`max_entities` is set (to 2). `no_bonus` stays `False` (default), so bonus
questions will fire for entities with rich text.

### 1c. Create the incremental cache

```python
# cli.py:155
cache = IncrementalCache(config.output_dir) if incremental else None
```

The constructor opens `.extraction_cache.json` from the output directory.
If it doesn't exist yet, the cache starts empty. The cache is a flat dict
keyed by `"{domain}_{entity_id}"`:

```python
{
    "compute-resources_ranch.tacc.access-ci.org": {
        "hash": "a1b2c3d4e5f6...",
        "pairs": [...]   # serialized QAPair dicts
    }
}
```

---

## Step 2: The Async Loop Dispatches Extractors

**File:** `cli.py:160-167`

```python
async def run_all():
    outputs = {}
    for server in servers:
        name, output = await run_extraction(server, config, incremental_cache=cache)
        outputs[name] = output
    return outputs

outputs = asyncio.run(run_all())
```

The extractors run **sequentially** — one domain at a time. Each call to
`run_extraction()` does:

```python
# cli.py:55-67
extractor_class = EXTRACTORS[server_name]
server_config = config.servers[server_name]
extraction_config = config.get_extraction_config(server_name)

extractor = extractor_class(
    server_config,
    extraction_config=extraction_config,
    incremental_cache=incremental_cache,
)

output = await extractor.run()
```

That `extractor.run()` call (cli.py:67, inside `run_extraction()`) is where
the two paths diverge. For compute-resources, `run()` is inherited from
`BaseExtractor`. For allocations, `run()` is overridden. Both eventually
call `self.extract()`, which is where the domain-specific logic lives.

---

## Step 3A: MCP Path — compute-resources

**File:** `extractors/base.py:68` → `extractors/compute_resources.py:58`

`run_extraction()` calls `await extractor.run()`. For compute-resources,
`run()` is **inherited from BaseExtractor**:

```python
# base.py:68
async def run(self) -> ExtractionOutput:
    async with MCPClient(self.config) as client:
        self.client = client
        return await self.extract()
```

This opens an HTTP connection to `http://localhost:3002` (the MCP server
running in Docker), stores it as `self.client`, then calls the abstract
`extract()` method which the subclass implements.

### 3A.1 Fetch all entities

```python
# compute_resources.py:64-65
result = await self.client.call_tool("search_resources", {"query": ""})
resources = result.get("resources", result.get("items", []))
```

`MCPClient.call_tool()` (mcp_client.py:27) POSTs to
`http://localhost:3002/tools/search_resources` with body
`{"arguments": {"query": ""}}`. The MCP server returns its envelope format:

```json
{"content": [{"type": "text", "text": "{\"total\": 23, \"items\": [...]}"}]}
```

`_parse_response()` (mcp_client.py:53) unwraps this — pulls the `text` field
from the first content block and `json.loads()` it. We get back a plain dict.
Line 65 extracts the items list (the key varies by server version).

Result: a list of ~23 HPC resource dicts.

### 3A.2 Build the system prompt (once, outside the loop)

```python
# compute_resources.py:67
system_prompt = build_system_prompt("compute-resources")
```

This is in `question_categories.py:214`. It calls
`format_categories_block("compute-resources")` which renders the 6 categories
(overview, organization, gpu_hardware, cpu_hardware, capabilities, access)
into a numbered list, then interpolates them into `SYSTEM_PROMPT_TEMPLATE`.

The system prompt tells the LLM: "You will generate Q&A pairs. Here are
the categories. Output a JSON array with `{category, question, answer}`."

This prompt is built **once** and reused for every entity in the domain.

### 3A.3 The per-entity loop

```python
# compute_resources.py:69-84
entity_count = 0
for resource in resources:
    resource_id = resource.get("id", "")
    resource_name = resource.get("name", "")
    if not resource_id or not resource_name:
        continue
    if "COMING SOON" in resource_name and not resource.get("description"):
        continue
    if self.extraction_config.max_entities is not None:
        if entity_count >= self.extraction_config.max_entities:
            break
    entity_count += 1
```

The loop iterates all resources but skips invalid entries and breaks at the
`max_entities` cap. For each of the 2 entities (capped by `--max-entities 2`):

**Fetch hardware details:**
```python
# compute_resources.py:87-94
hardware = {}
try:
    hw_result = await self.client.call_tool(
        "get_resource_hardware", {"id": resource_id}
    )
    hardware = hw_result
except Exception:
    pass
```

Another MCP call, same mechanics as above. Gets GPU counts, CPU specs, etc.
Wrapped in try/except so a missing hardware endpoint doesn't crash the whole run.

**Clean the data:**
```python
# compute_resources.py:97-103
clean_resource = self._clean_resource_data(resource)
clean_hardware = self._clean_hardware_data(hardware)

entity_data = {**clean_resource}
if clean_hardware:
    entity_data["hardware"] = clean_hardware
```

These private methods strip HTML tags, remove empty fields, normalize lists.
Hardware is nested under a `"hardware"` key rather than flat-merged.
The cleaned dict is what gets sent to the LLM and used for factoid templates.

**Compute the hash:**
```python
# compute_resources.py:106
entity_hash = compute_entity_hash(entity_data)
```

`compute_entity_hash()` (generators/incremental.py:15) does:
```python
json_str = json.dumps(data, sort_keys=True, default=str)
return hashlib.sha256(json_str.encode()).hexdigest()[:16]
```

Deterministic. If the same data comes back next run, same hash. If anything
changed (a description was updated, a GPU count changed), different hash.

### 3A.4 Incremental cache check

```python
# compute_resources.py:107-117
entity_hash = compute_entity_hash(entity_data)
used_cache = False
if self.incremental_cache:
    if self.incremental_cache.is_unchanged(
        "compute-resources", resource_id, entity_hash
    ):
        cached_pairs = self.incremental_cache.get_cached_pairs(
            "compute-resources", resource_id
        )
        if cached_pairs:
            pairs.extend(cached_pairs)
            used_cache = True
```

`is_unchanged()` (incremental.py:47) compares the new hash against the stored
hash. If they match, `get_cached_pairs()` (incremental.py:58) deserializes the
stored QAPair dicts back into `QAPair` objects via `QAPair.model_validate()`.
The `used_cache` flag gates the generation stages — if `True`, all three
stages (comprehensive, factoid, bonus) are skipped via `if not used_cache:`.

On first run, the cache is empty, so everything falls through to generation.

### 3A.5 Generation Stage 1: Comprehensive (LLM)

```python
# compute_resources.py:126
resource_pairs = await self._generate_qa_pairs(resource_id, entity_data, source_data, system_prompt)
```

`_generate_qa_pairs()` (compute_resources.py:246) does:

1. Build the user prompt:
   ```python
   user_prompt = build_user_prompt("compute-resources", resource_id, json.dumps(entity_data))
   ```
   This is `question_categories.py:224`. It renders `USER_PROMPT_TEMPLATE`,
   which says: "Here is the data for entity {id}. Generate Q&A pairs.
   End every answer with `<<SRC:compute-resources:{id}>>`."

2. Call the LLM:
   ```python
   response = self.llm.generate(
       system=system_prompt,
       user=user_prompt,
       max_tokens=self.extraction_config.max_tokens,
   )
   ```
   `self.llm` was created in `__init__` by `get_llm_client()` (llm_client.py).
   This factory reads `LLM_BACKEND` env var and returns the matching client
   (AnthropicClient, OpenAIClient, LocalLLMClient, or TransformersClient).
   All implement `.generate(system, user, max_tokens) -> LLMResponse`.
   Note: `generate()` is synchronous, not async.

3. Parse the response:
   ```python
   response_text = response.text
   json_match = re.search(r"\[[\s\S]*\]", response_text)
   if json_match:
       qa_list = json.loads(json_match.group())
   ```
   The LLM returns markdown with a JSON array embedded. The regex extracts it.

4. Create QAPairs:
   ```python
   for qa in qa_list:
       category = qa.get("category", "")
       question = qa.get("question", "")
       answer = qa.get("answer", "")

       if category and question and answer:
           pair_id = f"compute-resources_{resource_id}_{category}"

           complexity = "simple"
           if any(
               term in question.lower()
               for term in ["specifications", "how many", "performance", "compared"]
           ):
               complexity = "moderate"

           pairs.append(
               QAPair.create(
                   id=pair_id,
                   question=question,
                   answer=answer,
                   source_ref=f"mcp://compute-resources/resources/{resource_id}",
                   domain="compute-resources",
                   complexity=complexity,
                   source_data=source_data,
               )
           )
   ```
   `QAPair.create()` (models.py:46) wraps the Q&A into `Message` objects,
   auto-detects `<<SRC:` citations in the answer, and sets `granularity`
   to `"comprehensive"` (the default). Complexity defaults to `"simple"` and
   is upgraded to `"moderate"` if the question contains certain keywords.

Result: ~5 comprehensive QAPairs for this entity.

### 3A.6 Generation Stage 2: Factoid (Templates, No LLM)

```python
# compute_resources.py:132
factoid_pairs = generate_factoid_pairs("compute-resources", resource_id, entity_data)
```

`generate_factoid_pairs()` (generators/factoids.py:412) does:

1. Look up domain templates:
   ```python
   templates = FACTOID_TEMPLATES["compute-resources"]   # 7 templates
   ```
   Each template is a dict with `id`, `question`, `answer`, `required_fields`,
   and optionally `bool_field`. Example:

   ```python
   {
       "id": "resource_type",
       "question": "What type of resource is {name}?",
       "answer": "{name} is a {resource_type} resource.",
       "required_fields": ["name", "resource_type"],
   }
   ```

2. Prepare the entity data:
   ```python
   prepared = _prepare_compute_resources(entity_data)   # factoids.py:68
   ```
   This derives computed fields: joins `organization_names` into a string,
   formats GPU info, truncates descriptions. The **quality guards** live here —
   filtering out empty strings, "Unknown" values, recalculating counts after
   filtering.

3. Apply each template:
   ```python
   for template in templates:
       pair = _apply_template(template, prepared, domain, entity_id, citation, source_ref)
   ```
   `_apply_template()` (factoids.py:445) checks that all `required_fields`
   are present and truthy in `prepared`, then does `question.format(**prepared)`
   and `answer.format(**prepared)`.

   Before creating the QAPair, it runs the **post-format quality check**:
   ```python
   if _has_quality_defect(answer):   # factoids.py:49
       return None
   ```
   This catches interpolation artifacts:
   - `"Delta is operated by ."` (ends with ` .` — empty value before period)
   - `"has () GPUs"` (empty parenthetical)
   - `"NCSA, ."` (dangling comma)
   - Answers shorter than 10 characters

   If the answer passes, it appends the citation and creates:
   ```python
   QAPair.create(..., granularity="factoid")
   ```

Result: ~6 factoid QAPairs for this entity (some templates may be filtered
by missing data or quality defects).

### 3A.7 Generation Stage 3: Bonus (LLM)

```python
# compute_resources.py:138-143
bonus_pairs = []
if not self.extraction_config.no_bonus:
    bonus_pairs = generate_bonus_pairs(
        "compute-resources", resource_id, entity_data,
        self.llm, self.extraction_config.max_tokens,
    )
```

`no_bonus` is `False` (we didn't pass `--no-bonus`), so we enter
`generate_bonus_pairs()` (question_categories.py:307). This is the second
LLM call per entity. Here's the full call chain:

**Gate check — has_rich_text():**
```python
# question_categories.py:325
if not has_rich_text(domain, entity_data):
    return []
```

`has_rich_text()` (question_categories.py:285) looks up the rich text fields
for this domain:

```python
RICH_TEXT_FIELDS = {
    "compute-resources": ["description"],
    "software-discovery": ["description"],
    "allocations": ["abstract"],
    "nsf-awards": ["abstract"],
    "affinity-groups": ["description"],
}
```

For compute-resources, it checks `entity_data["description"]`. If that string
is shorter than 100 characters (or missing, or not a string), it returns `[]`
immediately — no LLM call. This is the gatekeeper. Most HPC systems have
substantial descriptions, so most pass.

**Build the bonus system prompt:**
```python
# question_categories.py:328
system_prompt = build_bonus_system_prompt("compute-resources")
```

`build_bonus_system_prompt()` (question_categories.py:295) renders
`BONUS_SYSTEM_PROMPT_TEMPLATE`. This prompt is different from the
comprehensive prompt. It tells the LLM:

1. Here are the categories already covered: overview, organization,
   gpu_hardware, cpu_hardware, capabilities, access
2. Do NOT repeat those topics
3. Find 1-3 additional questions about entity-unique details: notable
   collaborations, unusual capabilities, specific technologies, named
   methodologies, interdisciplinary aspects
4. If nothing unique exists beyond the standard categories, return `[]`
5. Cap at 3 questions maximum

**Build the user prompt (same function as comprehensive pass):**
```python
# question_categories.py:330
user_prompt = build_user_prompt("compute-resources", entity_id, json.dumps(entity_data))
```

Same `USER_PROMPT_TEMPLATE` as Stage 1. Same entity data. The system prompt
is the only thing that changed.

**Call the LLM:**
```python
# question_categories.py:332-337
response = llm_client.generate(
    system=system_prompt,
    user=user_prompt,
    max_tokens=max_tokens,
)
```

This is the **second** `llm.generate()` call for this entity (the first was
in `_generate_qa_pairs()` during Stage 1). Same LLM backend, same API.
Like the first call, `generate()` is synchronous.

**Parse the response:**
```python
# question_categories.py:339-343
json_match = re.search(r"\[[\s\S]*\]", response.text)
if not json_match:
    return []
qa_list = json.loads(json_match.group())
```

Same JSON extraction pattern as the comprehensive pass, with an early return
if the regex finds no match.

**Create QAPairs with sequential IDs:**
```python
# question_categories.py:344-368
pairs = []
bonus_num = 0
for qa in qa_list:
    if bonus_num >= 3:
        break
    question = qa.get("question", "")
    answer = qa.get("answer", "")
    if question and answer:
        bonus_num += 1
        pair_id = f"{domain}_{entity_id}_bonus_{bonus_num}"
        pattern = SOURCE_REF_PATTERNS.get(
            domain, f"mcp://{domain}/{entity_id}"
        )
        source_ref = pattern.format(entity_id=entity_id)

        pairs.append(
            QAPair.create(
                id=pair_id,
                question=question,
                answer=answer,
                source_ref=source_ref,
                domain=domain,
                granularity="exploratory",
            )
        )
return pairs
```

Key details:
- `bonus_num` is a counter, not an enumerate index. It only increments for
  valid pairs (non-empty question AND answer). So if the LLM returns an
  item with an empty question, it gets skipped and the next valid pair
  still gets `_bonus_1`, not `_bonus_2`.
- Hard cap at 3: `if bonus_num >= 3: break`.
- `granularity="exploratory"` — this is the only place in the codebase
  that creates exploratory pairs.
- `source_ref` uses the domain-specific pattern from `SOURCE_REF_PATTERNS`
  (question_categories.py:276). For compute-resources, this resolves to
  `mcp://compute-resources/resources/{entity_id}`.

**Error handling:**
```python
# question_categories.py:371-373
except Exception as e:
    print(f"Error generating bonus Q&A for {domain}/{entity_id}: {e}")
    return []
```

The entire function is wrapped in a try/except. If the LLM returns garbage,
the regex finds no match, or JSON parsing fails — we get 0 bonus pairs
instead of a crash (with a logged message). The comprehensive and factoid
pairs for this entity are unaffected.

Result: 0-3 exploratory QAPairs. For an HPC system with a rich description
(say, Delta's 400-word description of its GPU nodes and storage tiers),
you typically get 2-3 pairs about specific technologies or capabilities
that didn't fit neatly into the fixed categories.

### 3A.8 Generation Stage 4: Judge Evaluation (LLM)

```python
# compute_resources.py (after bonus, before cache store)
all_entity_pairs = resource_pairs + factoid_pairs + bonus_pairs
if self.judge_client:
    evaluate_pairs(all_entity_pairs, source_data, self.judge_client)
```

`evaluate_pairs()` (generators/judge.py) is the **third** LLM call per entity.
It sends ALL pairs for this entity (comprehensive + factoid + bonus) as a single
batch to a cheaper "judge" model (gpt-4o-mini or claude-haiku by default).

**Build the prompt:**

The judge receives a JSON block with each pair's `id`, `question`, and `answer`,
plus the full `source_data` (the raw entity data the answers were generated from).
The system prompt instructs it to score each pair on three dimensions (0.0-1.0):
- **Faithfulness** — does the answer accurately reflect the source data?
- **Relevance** — does the answer actually address the question?
- **Completeness** — does the answer cover the key facts from the source?

**Parse and apply scores:**

```python
# judge.py (inside evaluate_pairs)
json_match = re.search(r"\[[\s\S]*\]", response.text)
scores_list = json.loads(json_match.group())
scores_by_id = {s["pair_id"]: s for s in scores_list if "pair_id" in s}

for pair in pairs:
    scores = scores_by_id.get(pair.id)
    if not scores:
        continue
    faithfulness = float(scores.get("faithfulness", 0))
    relevance = float(scores.get("relevance", 0))
    completeness = float(scores.get("completeness", 0))
    confidence = min(faithfulness, relevance, completeness)

    pair.metadata.faithfulness_score = faithfulness
    pair.metadata.relevance_score = relevance
    pair.metadata.completeness_score = completeness
    pair.metadata.confidence_score = confidence
    pair.metadata.suggested_decision = (
        "approved" if confidence >= 0.8 else "needs_review"
    )
    pair.metadata.eval_issues = scores.get("issues", [])
```

Key details:
- **Confidence is deterministic** — computed in Python as `min(three scores)`, not
  by the LLM. The threshold (0.8) is a constant in `judge.py`.
- **Mutates metadata in-place** — the QAPair objects already exist. The judge just
  adds score fields to their `.metadata`. No new pairs are created.
- **Graceful degradation** — the entire function is wrapped in try/except. If the
  judge LLM fails or returns garbage, pairs keep their `None` scores and the
  pipeline continues. The comprehensive, factoid, and bonus pairs are unaffected.
- **Separate LLM client** — `self.judge_client` is created by `get_judge_client()`
  (llm_client.py), which reads `LLM_JUDGE_BACKEND` / `LLM_JUDGE_MODEL` env vars.
  Defaults to cheaper models than the generator.
- **Skippable** — `--no-judge` sets `self.judge_client = None`. Also, if no API key
  is available, the client init silently falls back to `None`.

Result: every QAPair for this entity now has `faithfulness_score`, `relevance_score`,
`completeness_score`, `confidence_score`, `suggested_decision`, and `eval_issues`
populated on its metadata. These scores flow into the cache and JSONL output.

### 3A.9 Store in cache

```python
# compute_resources.py:147-153
if self.incremental_cache:
    self.incremental_cache.store(
        "compute-resources", resource_id, entity_hash,
        resource_pairs + factoid_pairs + bonus_pairs,
    )
```

`store()` (incremental.py:67) serializes each QAPair via `.model_dump(mode="json")`
and stores it alongside the hash. Next run, if the hash matches, we skip
all three generation stages.

### 3A.10 Collect raw data for comparisons

```python
# compute_resources.py:156-164
raw_data[resource_id] = {
    "name": self._clean_name(resource_name),
    "resource_id": resource_id,
    "organizations": resource.get("organization_names", []),
    "has_gpu": resource.get("hasGpu", False),
    "gpu_types": self._extract_gpu_types(clean_hardware),
    "features": clean_resource.get("feature_names", []),
    "resource_type": resource.get("resourceType", ""),
}
```

This normalized dict is what `ComparisonGenerator` will use later to
group entities by shared attributes.

### 3A.11 Return

```python
return ExtractionOutput(pairs=pairs, raw_data=raw_data)
```

Back in `cli.py`, this output is appended to the `outputs` list.
On to the next domain.

---

## Step 3B: Direct API Path — allocations

**File:** `extractors/allocations.py:41` → `extractors/allocations.py:114`

For allocations, `run()` is **overridden**:

```python
# allocations.py:41
async def run(self) -> ExtractionOutput:
    return await self.extract()   # No MCPClient. No async with.
```

No `MCPClient` is created. No Docker container is contacted at `:3006`.
Instead, the extractor talks directly to the public allocations API.

### 3B.1 Fetch all entities via pagination

```python
# allocations.py:119
projects = await self._fetch_all_projects()
```

`_fetch_all_projects()` (allocations.py:73) creates its own httpx client
and paginates through `https://allocations.access-ci.org/current-projects.json`:

```python
# allocations.py:82-110
async with httpx.AsyncClient(timeout=30.0) as http:
    resp = await http.get(ALLOCATIONS_API_URL, params={"page": 1})
    resp.raise_for_status()
    data = resp.json()

    page_projects = data.get("projects", [])
    total_pages = data.get("pages", 1)
    all_projects.extend(page_projects)

    if max_entities and len(all_projects) >= max_entities:
        return all_projects[:max_entities]

    for page_num in range(2, total_pages + 1):
        resp = await http.get(ALLOCATIONS_API_URL, params={"page": page_num})
        resp.raise_for_status()
        page_projects = resp.json().get("projects", [])
        all_projects.extend(page_projects)

        if max_entities and len(all_projects) >= max_entities:
            return all_projects[:max_entities]
```

With `--max-entities 2`, it fetches page 1 (20 results) and immediately
returns the first 2. No additional pages needed.

### 3B.2 Everything else is the same

From here, the per-entity loop is structurally identical to the MCP path:

1. `_clean_project_data(project)` — domain-specific cleaning
2. `compute_entity_hash(clean_project)` — same function
3. Incremental cache check — same logic
4. `_generate_qa_pairs()` — same LLM call pattern, different prompt categories
5. `generate_factoid_pairs("allocations", ...)` — different templates, same function
6. `generate_bonus_pairs("allocations", ...)` — same bonus flow as 3A.7 above,
   but `has_rich_text()` checks `"abstract"` instead of `"description"`,
   and `source_ref` uses allocations API URI pattern
7. Cache store — same function
8. Raw data normalization — different fields, same pattern

The only structural difference is **no per-entity MCP calls**. In the MCP path,
compute-resources makes a second call per entity (`get_resource_hardware`).
In the direct API path, all entity data comes from the initial paginated fetch.

---

## Step 4: Back in the CLI — Comparisons

**File:** `cli.py:182-195` → `generators/comparisons.py:55`

After all extractors have run, the CLI collects their `raw_data` dicts:

```python
# cli.py:185-191
comparison_gen = ComparisonGenerator()
comparison_pairs = comparison_gen.generate(
    compute_data=outputs.get("compute-resources", ExtractionOutput([], {})).raw_data,
    software_data=outputs.get("software-discovery", ExtractionOutput([], {})).raw_data,
    allocations_data=outputs.get("allocations", ExtractionOutput([], {})).raw_data,
    nsf_awards_data=outputs.get("nsf-awards", ExtractionOutput([], {})).raw_data,
    affinity_groups_data=outputs.get("affinity-groups", ExtractionOutput([], {})).raw_data,
)
```

`ComparisonGenerator.generate()` (comparisons.py:55) runs ~11 sub-generators,
each one grouping entities by a shared attribute:

```
_generate_gpu_availability_questions()       → "Which resources have NVIDIA A100 GPUs?"
_generate_feature_availability_questions()   → "Which resources support interactive computing?"
_generate_organization_questions()           → "Which resources are operated by TACC?"
_generate_software_availability_questions()  → "Which resources have Python installed?"
_generate_allocations_by_fos()               → "Which projects are in the field of Physics?"
_generate_allocations_by_institution()       → "Which projects are from MIT?"
_generate_allocations_by_resource()          → "Which projects use Delta?"
_generate_nsf_by_program()                   → "Which NSF awards are from CISE?"
_generate_nsf_by_institution()               → "Which NSF awards are at Stanford?"
_generate_affinity_by_category()             → "Which affinity groups are in the Science category?"
```

Each creates QAPairs with `granularity="comparison"`. No LLM. The answers are
constructed programmatically from the entity lists with multi-entity citations.

Result: a handful of comparison pairs (depends on how many entities share
attributes in the 2-entity sample).

---

## Step 5: Save the Incremental Cache

```python
# cli.py:170-171
if cache:
    cache.save()
```

`save()` (incremental.py:77) writes the entire cache dict to
`.extraction_cache.json` in the output directory. Next run with
`--incremental`, entities with matching hashes will be replayed for free.

---

## Step 6: Write JSONL Output

**File:** `cli.py:223-231` → `output/jsonl_writer.py`

```python
# cli.py:223-231
writer = JSONLWriter(config.output_dir)

if combined:
    filepath = writer.write_combined(results)
else:
    filepaths = writer.write_all(results)
```

`write_all()` (jsonl_writer.py:45) iterates over each server name in `results`:

```python
# jsonl_writer.py:57-61
results = {}
for server_name, pairs in pairs_by_server.items():
    if pairs:
        results[server_name] = self.write(pairs, server_name=server_name)
return results
```

`write()` (jsonl_writer.py:16) opens a file named `{server_name}_qa_pairs.jsonl`
and writes one JSON line per QAPair:

```python
# jsonl_writer.py:38-41
with open(filepath, "w", encoding="utf-8") as f:
    for pair in pairs:
        line = pair.model_dump_json()
        f.write(line + "\n")
```

`model_dump_json()` is Pydantic v2's serializer. Each line is a complete
JSON object:

```json
{"id":"compute-resources_ranch.tacc.access-ci.org_overview","source":"mcp_extraction","source_ref":"mcp://compute-resources/resources/ranch.tacc.access-ci.org","domain":"compute-resources","messages":[{"role":"user","content":"What is Ranch and what is it designed for?"},{"role":"assistant","content":"Ranch is a storage system...<<SRC:compute-resources:ranch.tacc.access-ci.org>>"}],"metadata":{"complexity":"simple","granularity":"comprehensive","has_citation":true,"source_hash":null,"source_data":{...}}}
```

Output files for our invocation:
```
data/output/compute-resources_qa_pairs.jsonl   (comprehensive + factoid + exploratory)
data/output/allocations_qa_pairs.jsonl         (comprehensive + factoid + exploratory)
data/output/comparisons_qa_pairs.jsonl         (comparison)
```

All 4 granularities present. Exploratory pairs appear for entities whose
description/abstract was >= 100 chars.

---

## The Whole Trace in One Call Stack

```
YOU: qa-extract extract compute-resources allocations --max-entities 2 --incremental

cli.py:76  extract()
├── config.py               Config.from_env()                              # cli.py:129
├── cli.py:131-139          Apply --max-entities override
├── incremental.py:33       IncrementalCache(output_dir)  →  load .extraction_cache.json  # cli.py:155
│
├── cli.py:167              outputs = asyncio.run(run_all())
│   │
│   ├── cli.py:55           run_extraction("compute-resources", ...)
│   │   ├──                 ComputeResourcesExtractor.__init__()
│   │   │   └── llm_client.py  get_llm_client()  →  OpenAIClient (or whichever)
│   │   │
│   │   └── base.py:68      BaseExtractor.run()
│   │       ├── mcp_client.py:19   MCPClient.__aenter__()  →  open httpx to :3002
│   │       │
│   │       └── compute_resources.py:58  extract()
│   │           ├── mcp_client.py:27     call_tool("search_resources", {})  →  23 resources
│   │           ├── question_categories.py:214  build_system_prompt("compute-resources")
│   │           │
│   │           └── FOR EACH of 2 entities:
│   │               ├── mcp_client.py:27         call_tool("get_resource_hardware", {id})
│   │               ├── compute_resources.py:199 _clean_resource_data()
│   │               ├── compute_resources.py:222 _clean_hardware_data()
│   │               ├── incremental.py:15        compute_entity_hash()
│   │               ├── incremental.py:47        cache.is_unchanged()?  →  used_cache flag
│   │               │
│   │               ├── compute_resources.py:246 _generate_qa_pairs()          ← LLM call 1
│   │               │   ├── question_categories.py:224  build_user_prompt()
│   │               │   ├── llm_client.py                llm.generate()  →  LLM API call (sync)
│   │               │   ├── re.search + json.loads       parse JSON array
│   │               │   └── models.py:46                 QAPair.create() × ~5
│   │               │
│   │               ├── factoids.py:412          generate_factoid_pairs()       ← no LLM
│   │               │   ├── factoids.py:68       _prepare_compute_resources()
│   │               │   └── factoids.py:445      _apply_template() × 7
│   │               │       ├── factoids.py:49   _has_quality_defect()?
│   │               │       └── models.py:46     QAPair.create(granularity="factoid")
│   │               │
│   │               ├── question_categories.py:307  generate_bonus_pairs()     ← LLM call 2
│   │               │   ├── question_categories.py:285  has_rich_text()?
│   │               │   │   └── check entity_data["description"] >= 100 chars
│   │               │   ├── question_categories.py:295  build_bonus_system_prompt()
│   │               │   │   └── list covered categories, ask for 1-3 unique questions
│   │               │   ├── question_categories.py:224  build_user_prompt()  (same as pass 1)
│   │               │   ├── llm_client.py                llm.generate()  →  LLM API call (sync)
│   │               │   ├── re.search + json.loads       parse JSON array
│   │               │   └── models.py:46                 QAPair.create(granularity="exploratory")
│   │               │       └── bonus_num counter: skip empty items, cap at 3
│   │               │
│   │               ├── judge.py                   evaluate_pairs()              ← LLM call 3
│   │               │   ├── judge.py                build pairs block + source_data JSON
│   │               │   ├── llm_client.py           judge_client.generate()  →  cheaper model (sync)
│   │               │   ├── re.search + json.loads  parse scores array
│   │               │   └── mutate pair.metadata    faithfulness, relevance, completeness,
│   │               │                               confidence, suggested_decision, eval_issues
│   │               │
│   │               └── incremental.py:67        cache.store(hash, all_pairs + scores)
│   │
│   └── cli.py:55           run_extraction("allocations", ...)
│       ├──                 AllocationsExtractor.__init__()
│       │   └── llm_client.py  get_llm_client()
│       │
│       └── allocations.py:41  run()  ← OVERRIDES BaseExtractor.run()
│           │                           (no MCPClient created)
│           │
│           └── allocations.py:114  extract()
│               ├── allocations.py:73   _fetch_all_projects()
│               │   └── httpx.get(ALLOCATIONS_API_URL, params={"page": 1})  →  2 projects
│               ├── question_categories.py:214  build_system_prompt("allocations")
│               │
│               └── FOR EACH of 2 entities:
│                   ├── allocations.py:199       _clean_project_data()
│                   ├── incremental.py:15        compute_entity_hash()
│                   ├── incremental.py:47        cache.is_unchanged()?  →  used_cache flag
│                   │
│                   ├── allocations.py:231       _generate_qa_pairs()          ← LLM call 1
│                   │   ├── question_categories.py:224  build_user_prompt()
│                   │   ├── llm_client.py                llm.generate()  (sync)
│                   │   └── models.py:46                 QAPair.create() × ~5
│                   │
│                   ├── factoids.py:412          generate_factoid_pairs()       ← no LLM
│                   │   ├── factoids.py:124      _prepare_allocations()
│                   │   └── factoids.py:445      _apply_template() × 8
│                   │
│                   ├── question_categories.py:307  generate_bonus_pairs()     ← LLM call 2
│                   │   ├── question_categories.py:285  has_rich_text()?
│                   │   │   └── check entity_data["abstract"] >= 100 chars
│                   │   ├── question_categories.py:295  build_bonus_system_prompt()
│                   │   ├── question_categories.py:224  build_user_prompt()
│                   │   ├── llm_client.py                llm.generate()  (sync)
│                   │   └── models.py:46                 QAPair.create(granularity="exploratory")
│                   │
│                   ├── judge.py                   evaluate_pairs()              ← LLM call 3
│                   │   └── judge_client.generate()  →  cheaper model (sync)
│                   │
│                   └── incremental.py:67        cache.store(hash, all_pairs + scores)
│
├── incremental.py:77      cache.save()  →  write .extraction_cache.json     # cli.py:171
│
├── comparisons.py:55      ComparisonGenerator().generate(raw_data from both domains)  # cli.py:185
│   ├── comparisons.py:109 _generate_gpu_availability_questions()
│   ├── comparisons.py:145 _generate_feature_availability_questions()
│   ├── comparisons.py:190 _generate_organization_questions()
│   ├── comparisons.py:336 _generate_allocations_by_fos()
│   ├── comparisons.py:348 _generate_allocations_by_institution()
│   ├── comparisons.py:362 _generate_allocations_by_resource()
│   └── comparisons.py:613 _create_pair()  →  QAPair.create(granularity="comparison")
│
├── cli.py:203-216         Print Rich summary table
│
└── jsonl_writer.py:45     writer.write_all(results)                        # cli.py:229
    ├── write("compute-resources", pairs)  →  compute-resources_qa_pairs.jsonl
    ├── write("allocations", pairs)        →  allocations_qa_pairs.jsonl
    └── write("comparisons", pairs)        →  comparisons_qa_pairs.jsonl

DONE. All 4 granularities on disk. Up to 12 LLM calls total (3 per entity × 4 entities).
Cache primed for next run (including judge scores).
```

---

## What Each File Actually Does (In Execution Order)

| Order | File | What Happens There |
|---|---|---|
| 1 | `cli.py` | Parses CLI args, builds config, dispatches extractors, collects results, writes output |
| 2 | `config.py` | `Config.from_env()` reads env vars into dataclasses |
| 3 | `generators/incremental.py` | Loads/saves cache, hashes entities, compares hashes, replays cached pairs |
| 4 | `llm_client.py` | `get_llm_client()` factory → one of 4 backends, all with `.generate()` |
| 5 | `extractors/base.py` | `run()` opens MCPClient, calls `extract()`. Overridden by direct-API extractors |
| 6 | `mcp_client.py` | `call_tool()` POSTs to MCP server, unwraps response envelope |
| 7 | `extractors/compute_resources.py` | Fetches + cleans HPC resources, per-entity LLM + factoid + bonus loop |
| 8 | `extractors/allocations.py` | Overrides `run()`, paginates public API, same per-entity loop |
| 9 | `question_categories.py` | Defines categories, builds prompts, `generate_bonus_pairs()` |
| 10 | `generators/factoids.py` | Template-based Q&A, field preparation, quality guards |
| 11 | `generators/judge.py` | Judge LLM scores all pairs per entity (faithfulness, relevance, completeness) |
| 12 | `models.py` | `QAPair.create()` wraps Q&A into the canonical data model |
| 13 | `generators/comparisons.py` | Groups entities by shared attributes, creates cross-entity pairs |
| 14 | `output/jsonl_writer.py` | Writes `QAPair.model_dump_json()` lines to `.jsonl` files |

---

## Where the LLM Gets Called (Exactly Three Places)

All three live in the per-entity loop. All are skippable (cached entities skip all three).

1. **Comprehensive pass** — `_generate_qa_pairs()` inside each extractor.
   System prompt from `build_system_prompt()`. User prompt from `build_user_prompt()`.
   Always runs (unless entity is cached).

2. **Bonus pass** — `generate_bonus_pairs()` in `question_categories.py:307`.
   Different system prompt from `build_bonus_system_prompt()`. Same user prompt.
   Only runs if entity has rich text >= 100 chars AND `--no-bonus` is not set.

3. **Judge evaluation** — `evaluate_pairs()` in `generators/judge.py`.
   Sends all pairs for one entity as a batch. Uses a cheaper model (gpt-4o-mini
   or claude-haiku) via `get_judge_client()`. Scores each pair, doesn't create
   new ones. Skip with `--no-judge`.

That's it. Three `llm.generate()` calls per entity, max.

---

## Where Q&A Pairs Are Created (Exactly Four Places)

1. `_generate_qa_pairs()` → `QAPair.create(granularity="comprehensive")`
2. `_apply_template()` in factoids.py → `QAPair.create(granularity="factoid")`
3. `generate_bonus_pairs()` → `QAPair.create(granularity="exploratory")`
4. `_create_pair()` in comparisons.py → `QAPair.create(granularity="comparison")`

Every single QAPair in the system goes through `QAPair.create()` (models.py:46).
There is no other constructor path.
