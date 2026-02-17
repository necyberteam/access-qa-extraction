"""Command-line interface for Q&A extraction."""

import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load .env file from current directory or parent directories
load_dotenv()

from .citation_validator import CitationValidator
from .config import Config
from .extractors import (
    AffinityGroupsExtractor,
    AllocationsExtractor,
    ComputeResourcesExtractor,
    ExtractionOutput,
    ExtractionReport,
    NSFAwardsExtractor,
    SoftwareDiscoveryExtractor,
)
from .generators import ComparisonGenerator, IncrementalCache
from .models import ExtractionResult
from .output import JSONLWriter

app = typer.Typer(
    help="Extract Q&A pairs from ACCESS-CI MCP servers",
)
console = Console()


# Registry of available extractors
EXTRACTORS = {
    "compute-resources": ComputeResourcesExtractor,
    "software-discovery": SoftwareDiscoveryExtractor,
    "affinity-groups": AffinityGroupsExtractor,
    "allocations": AllocationsExtractor,
    "nsf-awards": NSFAwardsExtractor,
}


async def run_extraction(
    server_name: str,
    config: Config,
    incremental_cache: IncrementalCache | None = None,
) -> tuple[str, ExtractionOutput]:
    """Run extraction for a single server."""
    if server_name not in EXTRACTORS:
        console.print(f"[yellow]Skipping {server_name} - extractor not implemented[/yellow]")
        return server_name, ExtractionOutput(pairs=[], raw_data={})

    extractor_class = EXTRACTORS[server_name]
    server_config = config.servers[server_name]
    extraction_config = config.get_extraction_config(server_name)

    console.print(f"[blue]Extracting from {server_name}...[/blue]")
    extractor = extractor_class(
        server_config,
        extraction_config=extraction_config,
        incremental_cache=incremental_cache,
    )

    try:
        output = await extractor.run()
        console.print(f"[green]  Generated {len(output.pairs)} Q&A pairs[/green]")
        return server_name, output
    except Exception as e:
        console.print(f"[red]  Error: {e}[/red]")
        return server_name, ExtractionOutput(pairs=[], raw_data={})


@app.command()
def extract(
    servers: list[str] = typer.Argument(
        ..., help="Servers to extract from"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    combined: bool = typer.Option(
        False, "--combined", "-c", help="Write all pairs to a single file"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be extracted without writing"
    ),
    push_to_argilla: bool = typer.Option(
        False, "--push-to-argilla", help="Push extracted pairs to Argilla for review"
    ),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="Skip duplicate checking when pushing to Argilla"
    ),
    search_limit: int = typer.Option(
        None,
        "--search-limit",
        help="Max results per MCP query (overrides env/defaults). "
        "Only affects search-based extractors (software, allocations, nsf-awards).",
    ),
    max_queries: int = typer.Option(
        None,
        "--max-queries",
        help="How many search terms/queries to use from each extractor's list. "
        "Set low (1-2) for cheap test runs. None = use all.",
    ),
    max_entities: int = typer.Option(
        None,
        "--max-entities",
        help="Max entities to send to the LLM for Q&A generation. "
        "Applied after fetch and dedup. Works for all strategies. "
        "Set to 1 for a cheap single-entity test run.",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        "-i",
        help="Skip entities unchanged since last run (uses content hash).",
    ),
    no_bonus: bool = typer.Option(
        False,
        "--no-bonus",
        help="Skip LLM bonus question generation (faster, cheaper).",
    ),
):
    """Extract Q&A pairs from MCP servers."""
    config = Config.from_env()

    # CLI flags override env vars / defaults
    if search_limit is not None or max_queries is not None or max_entities is not None:
        for name in config.extraction:
            if search_limit is not None:
                config.extraction[name].search_limit = search_limit
            if max_queries is not None:
                config.extraction[name].max_queries = max_queries
            if max_entities is not None:
                config.extraction[name].max_entities = max_entities

    if no_bonus:
        for name in config.extraction:
            config.extraction[name].no_bonus = True

    if output:
        config.output_dir = str(output)

    # Validate server names
    for server in servers:
        if server not in config.servers:
            console.print(f"[red]Unknown server: {server}[/red]")
            raise typer.Exit(1)

    # Set up incremental cache if requested
    cache = IncrementalCache(config.output_dir) if incremental else None
    if cache:
        console.print("[blue]Incremental mode: skipping unchanged entities[/blue]")

    # Run extractions
    async def run_all():
        outputs = {}
        for server in servers:
            name, output = await run_extraction(server, config, incremental_cache=cache)
            outputs[name] = output
        return outputs

    outputs = asyncio.run(run_all())

    # Save cache and report stats
    if cache:
        cache.save()
        hits, misses = cache.stats
        console.print(
            f"[blue]Incremental: {hits} cached, {misses} regenerated[/blue]"
        )

    # Collect pairs and generate comparisons
    results: dict[str, ExtractionResult] = {}
    for name, output in outputs.items():
        if output.pairs:
            results[name] = output.pairs

    # Generate comparison Q&A pairs
    console.print("[blue]Generating comparison Q&A pairs...[/blue]")
    comparison_gen = ComparisonGenerator()
    comparison_pairs = comparison_gen.generate(
        compute_data=outputs.get("compute-resources", ExtractionOutput([], {})).raw_data,
        software_data=outputs.get("software-discovery", ExtractionOutput([], {})).raw_data,
        allocations_data=outputs.get("allocations", ExtractionOutput([], {})).raw_data,
        nsf_awards_data=outputs.get("nsf-awards", ExtractionOutput([], {})).raw_data,
        affinity_groups_data=outputs.get("affinity-groups", ExtractionOutput([], {})).raw_data,
    )
    if comparison_pairs:
        results["comparisons"] = comparison_pairs
        console.print(f"[green]  Generated {len(comparison_pairs)} comparison Q&A pairs[/green]")
    else:
        console.print("[yellow]  No comparison pairs generated[/yellow]")

    if not results:
        console.print("[yellow]No Q&A pairs generated[/yellow]")
        raise typer.Exit(0)

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

    if dry_run:
        console.print("\n[yellow]Dry run - no files written[/yellow]")
        return

    # Write output
    writer = JSONLWriter(config.output_dir)

    if combined:
        filepath = writer.write_combined(results)
        console.print(f"\n[green]Written to: {filepath}[/green]")
    else:
        filepaths = writer.write_all(results)
        console.print("\n[green]Written files:[/green]")
        for server_name, filepath in filepaths.items():
            console.print(f"  {server_name}: {filepath}")

    # Push to Argilla if requested
    if push_to_argilla:
        all_pairs = [p for pairs in results.values() for p in pairs]
        _push_pairs_to_argilla(all_pairs, check_duplicates=not no_dedup)


@app.command()
def report(
    servers: list[str] = typer.Argument(None, help="Servers to report on (omit for all)"),
    search_limit: int = typer.Option(
        None, "--search-limit", help="Override max results per MCP query.",
    ),
    max_queries: int = typer.Option(
        None, "--max-queries", help="Override how many queries to use.",
    ),
):
    """Show what each extractor would fetch from MCP (no LLM calls).

    Hits the MCP servers and reports how many entities each extractor
    finds at the current config settings. Useful for understanding
    coverage before spending LLM tokens.
    """
    config = Config.from_env()

    if search_limit is not None or max_queries is not None:
        for name in config.extraction:
            if search_limit is not None:
                config.extraction[name].search_limit = search_limit
            if max_queries is not None:
                config.extraction[name].max_queries = max_queries

    target_servers = servers or list(EXTRACTORS.keys())

    for server in target_servers:
        if server not in config.servers:
            console.print(f"[red]Unknown server: {server}[/red]")
            raise typer.Exit(1)

    async def run_reports() -> list[ExtractionReport]:
        reports = []
        for server_name in target_servers:
            if server_name not in EXTRACTORS:
                continue
            extractor_class = EXTRACTORS[server_name]
            server_config = config.servers[server_name]
            extraction_config = config.get_extraction_config(server_name)
            extractor = extractor_class(server_config, extraction_config=extraction_config)
            console.print(f"[blue]Checking {server_name}...[/blue]")
            try:
                r = await extractor.run_report()
                reports.append(r)
            except Exception as e:
                console.print(f"[red]  Error: {e}[/red]")
        return reports

    reports = asyncio.run(run_reports())

    if not reports:
        console.print("[yellow]No reports generated[/yellow]")
        raise typer.Exit(0)

    # Summary table
    table = Table(title="MCP Coverage Report")
    table.add_column("Server", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("Queries", justify="right")
    table.add_column("Fetched", justify="right")
    table.add_column("Unique", justify="right", style="green")
    table.add_column("Sample IDs")

    for r in reports:
        table.add_row(
            r.server_name,
            r.strategy,
            str(len(r.queries_used)) if r.queries_used else "-",
            str(r.total_fetched),
            str(r.unique_entities),
            ", ".join(r.sample_ids[:3]) + ("..." if len(r.sample_ids) > 3 else ""),
        )

    console.print(table)

    # Per-query breakdown for search-based extractors
    for r in reports:
        if r.queries_used:
            console.print(f"\n[bold]{r.server_name}[/bold] queries ({len(r.queries_used)}):")
            for q in r.queries_used:
                console.print(f"  - {q}")


def _push_pairs_to_argilla(pairs: list, check_duplicates: bool = True):
    """Push Q&A pairs to Argilla for review."""
    try:
        from .argilla_client import ArgillaClient
    except ImportError:
        console.print(
            "[red]Argilla dependencies not installed. "
            'Run: pip install -e ".[argilla]"[/red]'
        )
        raise typer.Exit(1)

    console.print("[blue]Pushing to Argilla...[/blue]")
    client = ArgillaClient()
    pushed, skipped = client.push_pairs(pairs, check_duplicates=check_duplicates)
    console.print(f"[green]  Pushed {pushed} records to Argilla[/green]")
    if skipped:
        console.print(f"[yellow]  Skipped {skipped} duplicates[/yellow]")


@app.command()
def push(
    file: Path = typer.Argument(..., help="JSONL file to push to Argilla"),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="Skip duplicate checking"
    ),
):
    """Push Q&A pairs from a JSONL file to Argilla for review."""
    from .output.jsonl_writer import load_jsonl

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    pairs = load_jsonl(file)
    console.print(f"[blue]Loaded {len(pairs)} Q&A pairs from {file}[/blue]")
    _push_pairs_to_argilla(pairs, check_duplicates=not no_dedup)


@app.command()
def list_servers():
    """List available MCP servers and their status."""
    config = Config.from_env()

    table = Table(title="MCP Servers")
    table.add_column("Server", style="cyan")
    table.add_column("URL", style="blue")
    table.add_column("Extractor", style="green")

    for name, server in config.servers.items():
        status = "✓ Implemented" if name in EXTRACTORS else "○ Not implemented"
        table.add_row(name, server.url, status)

    console.print(table)


@app.command()
def stats(
    file: Path = typer.Argument(..., help="JSONL file to analyze"),
):
    """Show statistics for a Q&A pairs file."""
    from .output.jsonl_writer import load_jsonl

    pairs = load_jsonl(file)

    # Domain breakdown
    domains: dict[str, int] = {}
    complexities: dict[str, int] = {}
    citations = 0

    for pair in pairs:
        domains[pair.domain] = domains.get(pair.domain, 0) + 1
        complexities[pair.metadata.complexity] = (
            complexities.get(pair.metadata.complexity, 0) + 1
        )
        if pair.metadata.has_citation:
            citations += 1

    console.print(f"\n[bold]File:[/bold] {file}")
    console.print(f"[bold]Total pairs:[/bold] {len(pairs)}")
    console.print(f"[bold]With citations:[/bold] {citations} ({100*citations//len(pairs)}%)")

    table = Table(title="By Domain")
    table.add_column("Domain", style="cyan")
    table.add_column("Count", justify="right", style="green")
    for domain, count in sorted(domains.items()):
        table.add_row(domain, str(count))
    console.print(table)

    table = Table(title="By Complexity")
    table.add_column("Complexity", style="cyan")
    table.add_column("Count", justify="right", style="green")
    for complexity, count in sorted(complexities.items()):
        table.add_row(complexity, str(count))
    console.print(table)


@app.command()
def validate(
    file: Path = typer.Argument(..., help="JSONL file to validate"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show all validation details"
    ),
):
    """Validate citations in Q&A pairs against MCP entities.

    Checks that all <<SRC:domain:entity_id>> citations reference
    real entities that exist in the corresponding MCP servers.
    """
    from .output.jsonl_writer import load_jsonl

    config = Config.from_env()
    pairs = load_jsonl(file)

    console.print(f"[blue]Validating {len(pairs)} Q&A pairs from {file}[/blue]")

    # Load entities and validate
    async def run_validation():
        validator = CitationValidator(config)
        await validator.load_entities()

        # Show loaded entities
        for domain in ["compute-resources", "software-discovery"]:
            entities = validator.get_known_entities(domain)
            console.print(f"  Loaded {len(entities)} {domain} entities")

        return validator

    validator = asyncio.run(run_validation())

    # Validate each pair
    valid_count = 0
    invalid_count = 0
    no_citation_count = 0
    invalid_pairs = []

    for pair in pairs:
        answer = pair.messages[1].content if len(pair.messages) > 1 else ""
        result = validator.validate_answer(answer)

        if not result.has_citations:
            no_citation_count += 1
            if verbose:
                console.print(f"[yellow]  {pair.id}: No citations[/yellow]")
        elif result.all_valid:
            valid_count += 1
            if verbose:
                console.print(f"[green]  {pair.id}: Valid[/green]")
        else:
            invalid_count += 1
            invalid_pairs.append((pair, result))

    # Show results
    console.print()
    table = Table(title="Validation Results")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")

    total = len(pairs)
    table.add_row("Valid citations", str(valid_count), f"{100*valid_count//total}%", style="green")
    table.add_row("Invalid citations", str(invalid_count), f"{100*invalid_count//total}%", style="red")
    table.add_row("No citations", str(no_citation_count), f"{100*no_citation_count//total}%", style="yellow")
    table.add_row("Total", str(total), "100%", style="bold")
    console.print(table)

    # Show invalid citations
    if invalid_pairs:
        console.print("\n[red]Invalid Citations:[/red]")
        for pair, result in invalid_pairs[:10]:  # Show first 10
            console.print(f"\n  [bold]{pair.id}[/bold]")
            question = pair.messages[0].content if pair.messages else "?"
            console.print(f"  Q: {question[:80]}...")
            for invalid in result.invalid_citations:
                console.print(f"  [red]✗ {invalid.citation.raw}[/red]")
                console.print(f"    {invalid.error}")

        if len(invalid_pairs) > 10:
            console.print(f"\n  ... and {len(invalid_pairs) - 10} more")

    # Exit with error if invalid citations found
    if invalid_count > 0:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
