#!/usr/bin/env python3
"""Run Q&A extraction on GH200 using local Qwen3 model.

This script runs the extraction with the transformers backend,
using the locally downloaded Qwen3-235B model.

Usage:
    python run_extraction_gh200.py

Environment variables:
    LLM_BACKEND=transformers  (default, uses local model)
    LOCAL_LLM_PATH=/home/apasquale/models/Qwen3-235B-A22B-Instruct-2507-FP8

The script will:
1. Load the Qwen3 model into GPU memory
2. Connect to the MCP servers (must be running)
3. Extract Q&A pairs from compute-resources and software-discovery
4. Generate comparison Q&A pairs
5. Write output to data/output/
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment for local LLM
os.environ["LLM_BACKEND"] = "transformers"
os.environ["LOCAL_LLM_PATH"] = "/home/apasquale/models/Qwen3-235B-A22B-Instruct-2507-FP8"

from rich.console import Console
from rich.table import Table

from access_qa_extraction.config import Config
from access_qa_extraction.extractors import ComputeResourcesExtractor, SoftwareDiscoveryExtractor, ExtractionOutput
from access_qa_extraction.generators import ComparisonGenerator
from access_qa_extraction.models import ExtractionResult
from access_qa_extraction.output import JSONLWriter
from access_qa_extraction.llm_client import get_llm_client

console = Console()


async def run_extraction(server_name: str, config: Config, llm_client):
    """Run extraction for a single server."""
    from access_qa_extraction.extractors import ComputeResourcesExtractor, SoftwareDiscoveryExtractor

    EXTRACTORS = {
        "compute-resources": ComputeResourcesExtractor,
        "software-discovery": SoftwareDiscoveryExtractor,
    }

    if server_name not in EXTRACTORS:
        console.print(f"[yellow]Skipping {server_name} - extractor not implemented[/yellow]")
        return server_name, ExtractionOutput(pairs=[], raw_data={})

    extractor_class = EXTRACTORS[server_name]
    server_config = config.servers[server_name]

    console.print(f"[blue]Extracting from {server_name}...[/blue]")
    extractor = extractor_class(server_config, llm_client=llm_client)

    try:
        output = await extractor.run()
        console.print(f"[green]  Generated {len(output.pairs)} Q&A pairs[/green]")
        return server_name, output
    except Exception as e:
        console.print(f"[red]  Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return server_name, ExtractionOutput(pairs=[], raw_data={})


async def main():
    console.print("[bold blue]ACCESS Q&A Extraction with Qwen3-235B[/bold blue]\n")

    # Load LLM client once (model will be cached)
    console.print("[blue]Loading Qwen3-235B model...[/blue]")
    llm_client = get_llm_client("transformers")
    console.print("[green]Model loaded![/green]\n")

    # Load config
    config = Config.from_env()
    config.output_dir = "data/output"

    # Ensure output directory exists
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Servers to extract from
    servers = ["compute-resources", "software-discovery"]

    # Run extractions
    outputs = {}
    for server in servers:
        name, output = await run_extraction(server, config, llm_client)
        outputs[name] = output

    # Collect pairs and generate comparisons
    results: dict[str, ExtractionResult] = {}
    for name, output in outputs.items():
        if output.pairs:
            results[name] = output.pairs

    # Generate comparison Q&A pairs
    console.print("\n[blue]Generating comparison Q&A pairs...[/blue]")
    comparison_gen = ComparisonGenerator()
    comparison_pairs = comparison_gen.generate(
        compute_data=outputs.get("compute-resources", ExtractionOutput([], {})).raw_data,
        software_data=outputs.get("software-discovery", ExtractionOutput([], {})).raw_data,
    )
    if comparison_pairs:
        results["comparisons"] = comparison_pairs
        console.print(f"[green]  Generated {len(comparison_pairs)} comparison Q&A pairs[/green]")
    else:
        console.print("[yellow]  No comparison pairs generated[/yellow]")

    if not results:
        console.print("[yellow]No Q&A pairs generated[/yellow]")
        return

    # Summary table
    table = Table(title="Extraction Summary")
    table.add_column("Server", style="cyan")
    table.add_column("Q&A Pairs", justify="right", style="green")
    table.add_column("Domains", style="magenta")

    total = 0
    for server_name, pairs in results.items():
        domains = set(p.domain for p in pairs)
        table.add_row(server_name, str(len(pairs)), ", ".join(sorted(domains)))
        total += len(pairs)

    table.add_row("Total", str(total), "", style="bold")
    console.print(table)

    # Write output
    writer = JSONLWriter(config.output_dir)
    filepaths = writer.write_all(results)
    console.print("\n[green]Written files:[/green]")
    for server_name, filepath in filepaths.items():
        console.print(f"  {server_name}: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())
