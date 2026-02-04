"""JSONL output writer for Q&A pairs."""

import json
from pathlib import Path

from ..models import ExtractionResult


class JSONLWriter:
    """Write Q&A pairs to JSONL files."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        pairs: ExtractionResult,
        filename: str | None = None,
        server_name: str | None = None,
    ) -> Path:
        """Write Q&A pairs to a JSONL file.

        Args:
            pairs: List of Q&A pairs to write
            filename: Optional filename (default: timestamped)
            server_name: Optional server name for filename prefix

        Returns:
            Path to the written file
        """
        if filename is None:
            prefix = f"{server_name}_" if server_name else ""
            filename = f"{prefix}qa_pairs.jsonl"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for pair in pairs:
                line = pair.model_dump_json()
                f.write(line + "\n")

        return filepath

    def write_all(
        self,
        pairs_by_server: dict[str, ExtractionResult],
    ) -> dict[str, Path]:
        """Write Q&A pairs from multiple servers to separate files.

        Args:
            pairs_by_server: Dict mapping server name to Q&A pairs

        Returns:
            Dict mapping server name to output file path
        """
        results = {}
        for server_name, pairs in pairs_by_server.items():
            if pairs:
                results[server_name] = self.write(pairs, server_name=server_name)
        return results

    def write_combined(
        self,
        pairs_by_server: dict[str, ExtractionResult],
        filename: str = "combined_qa_pairs.jsonl",
    ) -> Path:
        """Write all Q&A pairs to a single combined file.

        Args:
            pairs_by_server: Dict mapping server name to Q&A pairs
            filename: Output filename

        Returns:
            Path to the combined file
        """
        all_pairs = []
        for pairs in pairs_by_server.values():
            all_pairs.extend(pairs)

        return self.write(all_pairs, filename=filename)


def load_jsonl(filepath: str | Path) -> ExtractionResult:
    """Load Q&A pairs from a JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of Q&A pairs
    """
    from ..models import QAPair

    pairs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pairs.append(QAPair.model_validate(data))
    return pairs
