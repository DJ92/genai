# GenAI Research & Applications

> Production-ready GenAI implementations and research experiments

This repository contains practical GenAI applications and research experiments, demonstrating both breadth (RAG, agents, evaluation) and depth (implementation details, optimization).

## 🎯 Purpose

Bridge research → production for GenAI systems:
- **Production Patterns**: RAG, agents, tool use, guardrails
- **Optimization**: Latency, cost, quality trade-offs
- **Evaluation**: Automated metrics, LLM-as-judge, human eval
- **Real-world Use Cases**: Search, customer support, code generation

## 📂 Projects

### 1. Production RAG System
**Problem**: Build a retrieval-augmented generation system for enterprise knowledge bases

**Features**:
- Multiple chunking strategies (semantic, sentence, fixed-size)
- Hybrid search (embeddings + BM25)
- Reranking for quality
- Citation and source tracking
- Cost optimization

**Tech**: LangChain, ChromaDB, OpenAI/Anthropic

**Highlights**:
- Semantic chunking: 85% MRR@10
- Hybrid search: +12% over embeddings alone
- Latency: <500ms P99

---

### 2. Agentic Code Review System
**Problem**: Automated code review with LLM agents

**Capabilities**:
- Multi-step reasoning (ReAct pattern)
- Tool use (linter, test runner, git diff)
- Context management (relevant files only)
- Structured feedback generation

**Tech**: Anthropic Claude, GitHub API, AST parsing

**Highlights**:
- 78% precision on bug detection
- 85% recall on style violations
- Actionable feedback format

---

### 3. Multi-Agent Customer Support
**Problem**: Route and handle customer queries with specialized agents

**Architecture**:
- Router agent (intent classification)
- Specialist agents (billing, technical, general)
- Supervisor agent (quality control)
- Human handoff system

**Tech**: LangGraph, Claude, FastAPI

**Highlights**:
- 92% routing accuracy
- 3.2 avg turns to resolution
- 15% escalation rate

---

### 4. Prompt Optimization Pipeline
**Problem**: Systematically improve prompts with automated evaluation

**Features**:
- Prompt variants generation
- Automated evaluation (metrics + LLM-judge)
- Statistical significance testing
- Version control for prompts

**Tech**: DSPy, Anthropic, pytest

**Highlights**:
- 23% average quality improvement
- 5× faster iteration than manual
- Reproducible experiments

---

### 5. Guardrails & Safety System
**Problem**: Prevent harmful outputs and prompt injection attacks

**Components**:
- Input validation (prompt injection detection)
- Output moderation (toxicity, PII detection)
- Rate limiting and abuse prevention
- Monitoring and alerting

**Tech**: LlamaGuard, regex patterns, spaCy

**Highlights**:
- 94% injection detection precision
- 88% recall
- <50ms overhead per request

---

### 6. Semantic Code Search
**Problem**: Search codebase using natural language queries

**Features**:
- Code embedding generation (CodeBERT)
- Semantic search over functions/classes
- Context-aware snippets
- Multi-language support (Python, JS, Go)

**Tech**: CodeBERT, FAISS, tree-sitter

**Highlights**:
- 82% relevance @ top-5
- Sub-second search on 100k functions
- AST-aware chunking

---

### 7. Document Intelligence Pipeline
**Problem**: Extract structured data from unstructured documents (PDFs, images)

**Pipeline**:
- OCR (Tesseract, Google Vision)
- Layout analysis
- Entity extraction (NER)
- Structured output generation

**Tech**: Claude Vision, Donut, LayoutLM

**Highlights**:
- 95% entity extraction accuracy
- Multi-page PDF support
- Table extraction with structure

---

### 8. LLM Cost Optimizer
**Problem**: Reduce API costs without sacrificing quality

**Techniques**:
- Prompt compression (remove redundancy)
- Response caching (semantic similarity)
- Model routing (GPT-4 vs GPT-3.5)
- Batch processing

**Tech**: OpenAI API, Redis, prompt optimization

**Highlights**:
- 65% cost reduction
- <2% quality degradation
- Smart caching (90% hit rate)

---

### 9. Multimodal RAG System
**Problem**: Search and answer questions over documents with text AND images

**Features**:
- CLIP-based image embeddings
- Unified text + image retrieval
- Vision-language model integration (GPT-4V, Claude 3)
- Multi-modal reranking
- Image captioning for context

**Tech**: CLIP, GPT-4V/Claude 3 Vision, FAISS, OCR

**Highlights**:
- 88% retrieval accuracy (text+image)
- Handles PDFs, slides, diagrams
- Image-aware question answering

---

## 🛠 Common Patterns

### RAG Architecture
```
User Query
    ↓
1. Query Rewriting (improve retrieval)
    ↓
2. Hybrid Retrieval (embeddings + BM25)
    ↓
3. Reranking (top-k → top-5)
    ↓
4. Context Compression (fit in context window)
    ↓
5. LLM Generation (with citations)
    ↓
6. Verification (hallucination check)
```

### Agent Architecture (ReAct)
```
Task
    ↓
Loop until solved:
    1. Thought: What should I do next?
    2. Action: Execute tool/API call
    3. Observation: Get result
    4. [Repeat or Finish]
    ↓
Final Answer
```

### Guardrails Pattern
```
User Input
    ↓
Input Validation
    ↓ (safe)
LLM Processing
    ↓
Output Moderation
    ↓ (safe)
Return to User
```

---

## 📊 Best Practices & Patterns

### 1. Prompt Engineering
```python
# ❌ Bad: Vague, no structure
prompt = "Summarize this document"

# ✅ Good: Clear, structured, examples
prompt = """
Task: Summarize the key points from the document below.

Instructions:
1. Extract 3-5 main points
2. Keep each point to 1 sentence
3. Focus on actionable insights

Format:
- Point 1: ...
- Point 2: ...
- Point 3: ...

Document:
{document}

Summary:
"""
```

### 2. Error Handling with Retries
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm_with_retry(prompt):
    try:
        return llm.complete(prompt)
    except RateLimitError:
        logger.warning("Rate limited, retrying...")
        raise
```

### 3. Cost Tracking
```python
class CostTracker:
    def __init__(self):
        self.total_cost = 0
        self.request_count = 0

    def track_request(self, model, input_tokens, output_tokens):
        cost = calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.request_count += 1
```

---

## 📈 Optimization Strategies

### Latency Optimization
1. **Streaming**: Return partial results as they arrive
2. **Caching**: Cache common queries (semantic similarity)
3. **Parallel calls**: Batch independent LLM calls
4. **Prompt compression**: Remove unnecessary tokens
5. **Model selection**: Use smaller models when possible

### Cost Optimization
1. **Prompt optimization**: Shorter prompts, same quality
2. **Response caching**: 90%+ hit rate on common queries
3. **Model routing**: GPT-4 only when necessary
4. **Batch processing**: Aggregate requests
5. **Output length limits**: Prevent verbose responses

### Quality Optimization
1. **Few-shot examples**: 3-5 examples for consistency
2. **Chain-of-thought**: Step-by-step reasoning for complex tasks
3. **Self-consistency**: Sample multiple times, take majority
4. **Critique-and-revise**: Two-pass generation
5. **Ensemble**: Combine multiple model outputs

---

## 📚 Resources

### Papers Implemented
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

### Related Projects
- [AI Research Portfolio](https://github.com/DJ92/ai-research-portfolio) - Deep dives into LLM internals
- [ML System Design](https://github.com/DJ92/ml-system-design) - Production ML system architectures
- [Blog](https://dj92.github.io/interview-notes) - Technical writing on GenAI topics

---

## 📫 Contact

- Email: joshidheeraj1992@gmail.com
- GitHub: [@DJ92](https://github.com/DJ92)
- Blog: [dj92.github.io/interview-notes](https://dj92.github.io/interview-notes)

---

*Demonstrating production GenAI expertise through practical implementations and research experiments.*
