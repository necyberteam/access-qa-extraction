# Hallucination Mitigation Plan

This content has been integrated into the main planning documentation:

- **[access-qa-planning/04-model-training.md](https://github.com/necyberteam/access-qa-planning/blob/main/04-model-training.md)** - Hallucination detection, fallback architecture, Q&A RAG
- **[access-qa-planning/02-training-data.md](https://github.com/necyberteam/access-qa-planning/blob/main/02-training-data.md)** - Dual use of Q&A corpus (training + retrieval)

## Quick Reference

### Architecture

```
User Question
      │
      ▼
Fine-tuned Model (~200ms)
      │
      ▼
Citation Validation (check against MCP entities)
      │
      ├── Valid → Return response (fast path)
      │
      └── Invalid → Q&A RAG Fallback
                          │
                          ├── Match found → Return retrieved answer
                          └── No match → Graceful refusal
```

### Key Findings

1. **Fine-tuned models hallucinate** - They learn citation format but fabricate citations for unknown topics
2. **Citation validation works** - 100% detection rate by checking citations against MCP entities
3. **Single corpus** - Same Q&A pairs power both fine-tuning and RAG retrieval
4. **Q&A RAG is proven** - FAQ-style retrieval is simpler and faster than document RAG
