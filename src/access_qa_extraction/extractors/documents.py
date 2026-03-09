"""Extractor for document files (PDF, DOCX, TXT, MD).

Reads files from a local directory instead of calling MCP tools.
Uses the same two-shot LLM pipeline (battery + discovery) as MCP extractors,
with document-specific field guidance.

Each document (or chunk of a large document) is treated as one entity.
"""

import json
import re
from pathlib import Path

from ..generators.incremental import compute_entity_hash
from ..generators.judge import evaluate_pairs
from ..llm_client import BaseLLMClient, get_judge_client, get_llm_client
from ..models import ExtractionResult, QAPair
from ..parsers import (
    SUPPORTED_EXTENSIONS,
    chunk_text,
    clean_extracted_text,
    parse_document,
)
from ..question_categories import (
    build_battery_system_prompt,
    build_discovery_system_prompt,
    build_user_prompt,
)
from .base import BaseExtractor, ExtractionOutput


class DocumentExtractor(BaseExtractor):
    """Extract Q&A pairs from document files."""

    server_name = "documents"

    def __init__(self, *args, llm_client: BaseLLMClient | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_client or get_llm_client()
        self.judge_client = None
        if not self.extraction_config.no_judge:
            try:
                self.judge_client = get_judge_client()
            except (ValueError, ImportError):
                pass

    async def run(self) -> ExtractionOutput:
        """Run extraction — no MCPClient needed for documents."""
        return await self.extract()

    async def extract(self) -> ExtractionOutput:
        """Extract Q&A pairs from all documents in the configured directory."""
        pairs: ExtractionResult = []
        raw_data: dict = {}

        docs_dir = Path(self.config.url)
        if not docs_dir.is_absolute():
            # Resolve relative to the current working directory
            docs_dir = Path.cwd() / docs_dir
        docs_dir = docs_dir.resolve()

        if not docs_dir.exists():
            print(f"Documents directory not found: {docs_dir}")
            return ExtractionOutput(pairs=pairs, raw_data=raw_data)

        # Discover document files recursively
        doc_files = sorted(
            f for f in docs_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not doc_files:
            print(f"No supported documents found in {docs_dir}")
            return ExtractionOutput(pairs=pairs, raw_data=raw_data)

        system_prompt = build_battery_system_prompt("documents")

        entity_count = 0
        for doc_path in doc_files:
            # Build entity ID from filename stem
            base_entity_id = doc_path.stem

            # Filter to specific entity IDs if requested
            if self.extraction_config.entity_ids is not None:
                if base_entity_id not in self.extraction_config.entity_ids:
                    continue

            # Respect max_entities limit
            if self.extraction_config.max_entities is not None:
                if entity_count >= self.extraction_config.max_entities:
                    break

            # Parse document
            try:
                raw_text = parse_document(doc_path)
            except Exception as e:
                print(f"Error parsing {doc_path.name}: {e}")
                continue

            if not raw_text.strip():
                print(f"Skipping empty document: {doc_path.name}")
                continue

            text = clean_extracted_text(raw_text)

            # Chunk if needed
            chunks = chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    entity_id = f"{base_entity_id}__chunk_{chunk_idx + 1}"
                else:
                    entity_id = base_entity_id

                entity_count += 1
                if self.extraction_config.max_entities is not None:
                    if entity_count > self.extraction_config.max_entities:
                        break

                # Build entity data for LLM
                entity_data = {
                    "title": _title_from_stem(base_entity_id),
                    "source_file": doc_path.name,
                    "content": chunk,
                }

                # Check incremental cache
                entity_hash = compute_entity_hash(entity_data)
                used_cache = False
                if self.incremental_cache:
                    if self.incremental_cache.is_unchanged(
                        "documents", entity_id, entity_hash
                    ):
                        cached_pairs = self.incremental_cache.get_cached_pairs(
                            "documents", entity_id
                        )
                        if cached_pairs:
                            pairs.extend(cached_pairs)
                            used_cache = True

                if not used_cache:
                    source_data = {
                        "file": doc_path.name,
                        "content_preview": chunk[:500],
                    }

                    doc_pairs = await self._generate_qa_pairs(
                        entity_id, entity_data, source_data, system_prompt
                    )
                    pairs.extend(doc_pairs)

                    if self.judge_client:
                        evaluate_pairs(doc_pairs, source_data, self.judge_client)

                    if self.incremental_cache:
                        self.incremental_cache.store(
                            "documents", entity_id, entity_hash, doc_pairs
                        )

                raw_data[entity_id] = {
                    "file": doc_path.name,
                    "title": _title_from_stem(base_entity_id),
                    "word_count": len(chunk.split()),
                    "chunk": chunk_idx + 1 if len(chunks) > 1 else None,
                    "total_chunks": len(chunks) if len(chunks) > 1 else None,
                }

        return ExtractionOutput(pairs=pairs, raw_data=raw_data)

    async def _generate_qa_pairs(
        self,
        entity_id: str,
        entity_data: dict,
        source_data: dict,
        system_prompt: str,
    ) -> ExtractionResult:
        """Use LLM to generate Q&A pairs from document content."""
        pairs: ExtractionResult = []

        entity_json = json.dumps(entity_data, indent=2)
        entity_name = entity_data.get("title", entity_id)
        user_prompt = build_user_prompt(
            "documents", entity_id, entity_json, entity_name=entity_name
        )

        try:
            # Battery pass
            response = self.llm.generate(
                system=system_prompt,
                user=user_prompt,
                max_tokens=self.extraction_config.max_tokens,
            )
            qa_list = _parse_qa_response(response.text)

            # Discovery pass
            if qa_list:
                existing = [{"question": qa["question"], "answer": qa["answer"]} for qa in qa_list]
                discovery_prompt = build_discovery_system_prompt("documents", existing)
                discovery_response = self.llm.generate(
                    system=discovery_prompt,
                    user=user_prompt,
                    max_tokens=self.extraction_config.max_tokens,
                )
                qa_list.extend(_parse_qa_response(discovery_response.text))

            for seq_n, qa in enumerate(qa_list, start=1):
                question = qa.get("question", "")
                answer = qa.get("answer", "")

                if question and answer:
                    pair_id = f"documents_{entity_id}_{seq_n}"
                    pairs.append(
                        QAPair.create(
                            id=pair_id,
                            question=question,
                            answer=answer,
                            source_ref=f"doc://documents/{entity_id}",
                            domain="documents",
                            source_data=source_data,
                            source="doc_generated",
                        )
                    )

        except Exception as e:
            print(f"Error generating Q&A for {entity_id}: {e}")

        return pairs


def _title_from_stem(stem: str) -> str:
    """Convert a filename stem to a human-readable title.

    Examples:
        "data:allocation-duration-and-forfeiture" -> "Allocation Duration and Forfeiture"
        "using-jupyter-notebooks-on-bridges-2" -> "Using Jupyter Notebooks on Bridges 2"
        "10_1758119706.911465_data-ACCESS-how-to-cite-Jetstream" -> "How To Cite Jetstream"
    """
    # Strip Slack-style numeric prefixes (e.g., "10_1758119706.911465_")
    stem = re.sub(r"^\d+_[\d.]+_", "", stem)
    # Strip common prefixes (longest first to avoid partial matches)
    for prefix in ("data-ACCESS-", "data:ACCESS-", "data-", "data:", "ACCESS-"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    # Replace hyphens/underscores with spaces and title-case
    title = stem.replace("-", " ").replace("_", " ")
    return title.title()


def _parse_qa_response(response_text: str) -> list[dict]:
    """Parse a JSON array of Q&A pairs from an LLM response."""
    json_match = re.search(r"\[[\s\S]*\]", response_text)
    if json_match:
        return json.loads(json_match.group())
    return []
