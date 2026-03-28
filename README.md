# PromptDescent

## A Production Implementation of OPRO: Large Language Models as Optimizers

### Wei et al., Google DeepMind, 2023 | arXiv:2309.03409

[![Implementation Status](https://img.shields.io/badge/Status-Phase%201%20%E2%80%94%20Results%20Verified-brightgreen)](https://github.com/datadave22/PromptDescent)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](https://github.com/datadave22/PromptDescent)
[![Financial Δ](https://img.shields.io/badge/Financial%20%CE%94Score-%2B66.7pp-brightgreen)](https://github.com/datadave22/PromptDescent)
[![Model](https://img.shields.io/badge/Model-claude--sonnet--4--6-blue)](https://www.anthropic.com)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2309.03409-red)](https://arxiv.org/abs/2309.03409)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Executive Summary

PromptDescent is a full-stack research implementation that uses large language models to automatically optimize natural language prompts — no gradients, no backpropagation, no hand-tuning. It implements the OPRO algorithm directly from the paper, treating prompt optimization as black-box search over a discrete semantic space where standard calculus-based methods are fundamentally inapplicable. In production AI systems, the quality of a prompt directly determines output quality; PromptDescent makes that optimization process principled, reproducible, and observable.

---

## Why This Matters for Production AI

### The Production Prompt Engineering Problem

In production LLM systems, prompt quality is the primary lever for output quality — more impactful than model choice for most classification and extraction tasks. Manual prompt engineering is expensive, inconsistent across team members, non-reproducible, and fails to generalize across domains. A prompt optimized for one dataset degrades on distribution shift. PromptDescent makes this process algorithmic: define the task, define the evaluation set, run the optimizer.

### Why This Is an Engineering Problem, Not Just a Research Problem

The OPRO algorithm requires: an async streaming architecture to surface iteration-level results without blocking, a persistent trajectory store for auditability and reproducibility, a scoring harness that evaluates candidates in parallel without cross-contamination, and a UI that makes the optimization process observable in real time. These are production engineering requirements. This implementation addresses all four.

### Quant Finance Applications

The same algorithm and infrastructure applies across the full range of systematic finance NLP tasks:

- **Earnings call sentiment:** classify transcript segments as bullish/bearish/neutral for systematic equity signal generation
- **Credit event prediction:** classify Fed statements, bank filings, and credit rating agency language for fixed income strategies
- **M&A signal extraction:** classify merger announcement language for event-driven strategies
- **Regulatory filing analysis:** classify 10-K/10-Q risk factor language for portfolio risk monitoring
- **Macro regime classification:** classify central bank communication for rates and FX strategies

Define the labeled evaluation set. Run PromptDescent. The optimizer finds the prompt.

---

## The Mathematical Framework

### Problem Statement

Given an evaluation function mapping natural language prompts to task performance scores:

```
f: P → [0, 1]
```

where P is the discrete space of all natural language token sequences, find the optimal prompt:

```
p* = argmax_{p ∈ P} f(p)
```

The objective function f is defined as exact-match accuracy over a fixed evaluation set E:

```
f(p) = (1/|E|) Σ_{e ∈ E} 1[ LLM_evaluator(p, e.input) == e.expected ]
```

This is a black-box discrete optimization problem. f is expensive to evaluate (each call requires |E| LLM inference passes), non-differentiable by construction, and operates over an astronomically large search space with no known topology.

---

### Why Gradient Methods Fail

Standard first-order optimization requires:

```
p_{t+1} = p_t - η ∇f(p_t)
```

This requires ∂f/∂p to exist. For discrete token sequences, this derivative is **undefined**:

- p ∈ P is a sequence of discrete integer token IDs, not a point in ℝⁿ
- There is no continuous relaxation assumed — we are not computing soft token distributions
- Backpropagation through prompt text would require differentiable token embeddings **and** end-to-end training through both the optimizer and evaluator LLMs
- OPRO explicitly avoids this: the optimizer LLM and evaluator LLM are treated as black boxes accessible only via inference API

Even gradient-surrogate methods (straight-through estimators, Gumbel-softmax relaxations) require control over the forward pass internals. When the LLM is an external API call, no such access exists. The gradient simply does not exist in the operational context.

---

### The OPRO Algorithm

**State:** Maintain a trajectory of all evaluated (prompt, score) pairs, sorted ascending by score:

```
τ_t = { (p_1, s_1), (p_2, s_2), ..., (p_t, s_t) }   where s_1 ≤ s_2 ≤ ... ≤ s_t
```

**Meta-prompt construction:** At iteration t, construct M(τ_t) from three components:

```
M(τ_t) = [task_description]
        + [trajectory formatted as scored examples, ascending by score]
        + [generation instruction: "generate a better prompt"]
```

**Update rule:** Sample N candidate prompts from the optimizer LLM conditioned on the meta-prompt:

```
{ p^(1)_{t+1}, ..., p^(N)_{t+1} } ~ LLM_optimizer( M(τ_t) )
```

**Scoring:** Evaluate each candidate via the objective function (parallelizable):

```
s^(i)_{t+1} = f(p^(i)_{t+1}) = (1/|E|) Σ_{e ∈ E} 1[ LLM_evaluator(p^(i)_{t+1}, e.input) == e.expected ]
```

**Trajectory update:** Append the best candidate from this iteration:

```
τ_{t+1} = τ_t ∪ { (p*_{t+1}, s*_{t+1}) }
    where p*_{t+1} = argmax_i f(p^(i)_{t+1})
```

**Convergence intuition:** As |τ_t| grows, LLM_optimizer receives increasing signal about the shape of f over P. The ascending sort exposes the direction of improvement — prompts near the top of the list are better. This is functionally analogous to providing sign(∂f/∂p) in gradient descent, but operating in semantic space rather than parameter space. The LLM's parametric knowledge acts as an implicit prior over P, enabling sample-efficient search.

Formally, let Δs_t = s_t - s_{t-1} represent the score improvement at iteration t. The ascending-sorted trajectory presents the LLM with an implicit finite-difference approximation of the objective landscape: prompts with low scores occupy the bottom of the context, prompts with high scores occupy the top, and the generation instruction asks for movement toward higher scores. This is structurally analogous to the finite-difference gradient estimate used in zeroth-order optimization: ∇f(x) ≈ (f(x + δ) - f(x)) / δ — except the "perturbation" is semantic (a new prompt variant) rather than numerical, and the "gradient" is communicated implicitly through the ranked trajectory rather than computed explicitly. OPRO is zeroth-order optimization operating in semantic space.

---

### Connection to Gradient-Free Optimization Methods

| Method | Search Space | Direction Signal | Update Mechanism | Convergence Guarantee |
|---|---|---|---|---|
| Gradient Descent | ℝⁿ (continuous) | ∇f(x) | x_{t+1} = x_t - η∇f(x_t) | Local minimum under L-smoothness |
| Bayesian Optimization | ℝⁿ (continuous) | Acquisition function α(x) | Surrogate posterior q(f\|τ_t) | Global with enough samples |
| CMA-ES | ℝⁿ (continuous) | Covariance matrix Σ_t | Sample from N(μ_t, σ²_t Σ_t) | Global for unimodal objectives |
| Simulated Annealing | Discrete or continuous | Acceptance probability P(ΔE, T) | Stochastic neighbor transition | Global with annealing schedule |
| Zeroth-Order / Finite Difference | ℝⁿ (continuous) | (f(x+δ)-f(x))/δ ≈ ∇f(x) | x_{t+1} = x_t - η·∇̂f(x_t) | Local minimum, noisy gradient |
| **OPRO (this paper)** | **P (discrete NL)** | **Score trajectory τ_t** | **p_{t+1} ~ LLM(M(τ_t))** | **Empirical — no formal guarantee** |

OPRO occupies a previously empty cell in this table: a principled search algorithm for discrete natural language space that requires no internal access to the models being used. Every other method either assumes a continuous domain or requires explicit neighborhood structure. OPRO gets direction signal from the in-context learning capacity of the optimizer LLM itself.

---

### Connection to Bayesian Optimization

The structural parallel between OPRO and Bayesian Optimization is precise:

**In Bayesian Optimization:**
- Surrogate model: `q(f | τ_t)` — a Gaussian Process approximating the true objective, fit to observed (x, f(x)) pairs
- Acquisition function: `α(x) = E[improvement | q(f | τ_t)]` — guides where to query next, trading exploration vs. exploitation
- Next query: `x_{t+1} = argmax_x α(x)`

**In OPRO:**
- Implicit surrogate: `LLM_optimizer` implicitly models `q(f | τ_t)` via in-context learning — the trajectory τ_t serves as the "training set" for a few-shot learned surrogate
- Acquisition signal: `M(τ_t)` — the ascending-sorted trajectory with generation instruction serves as the acquisition function, directing the LLM toward regions of P with higher f
- Next query: `p_{t+1} ~ LLM_optimizer(M(τ_t))` — stochastic rather than deterministic

Both methods are strategies for **sample-efficient search in expensive-to-evaluate objective landscapes**. The key distinctions:
- BO operates in continuous ℝⁿ with an explicit probabilistic model (GP); OPRO operates in discrete semantic space using the LLM's parametric knowledge as an implicit prior
- BO's acquisition function is analytically derived from the surrogate; OPRO's "acquisition" is emergent from the LLM's instruction-following and in-context learning
- BO has theoretical guarantees under GP assumptions; OPRO has empirical results without formal convergence proofs

For quant researchers: f(p) is directly analogous to a backtest performance metric — a black-box function expensive to evaluate (each evaluation requires |E| model inference calls), motivating exactly the sample-efficient search strategy OPRO provides.

---

## Financial Applications

The second benchmark (`test-financial.ts`) applies OPRO to **earnings call sentiment classification** — bullish / bearish / neutral — a production NLP task in quantitative finance.

**Why this matters:** Earnings call transcripts are among the highest-signal, highest-frequency data sources in systematic equity strategies. Manually engineering prompts for financial NLP is expensive, inconsistent across analysts, and fails to generalize across asset classes, sectors, and market regimes. A prompt that classifies technology sector earnings calls well may fail on energy sector language. OPRO provides a principled, reproducible optimization procedure: define the evaluation set, specify the task, run the algorithm.

**Practical implications:**
- Prompt optimization can be rerun on new labeled data without human re-engineering
- The trajectory log τ_t is a full audit trail of the search — useful for model governance and compliance documentation
- Scores are exact-match accuracy over a fixed evaluation set, making results reproducible and comparable across runs
- The same algorithm generalizes to any classification task: credit rating action prediction, M&A outcome classification, regulatory filing sentiment

---

## Experimental Results

| Task | Domain | Initial Score | Final Score | Δ Score | Iterations |
|---|---|---|---|---|---|
| Movie Review Sentiment | NLP benchmark | 100.0% | 100.0% | +0.0pp | 5 |
| Earnings Call Sentiment | Quantitative finance | 33.3% | 100.0% | +66.7pp | 5 |

> The sentiment benchmark demonstrates OPRO's correct behavior at ceiling — when f(p_initial) = 1.0, the algorithm correctly produces no degradation. The financial benchmark demonstrates OPRO's primary use case: f(p_initial) = 0.333 on domain-specific 3-class classification, optimized to f(p*) = 1.0 in a single iteration. The optimized prompt spontaneously produced role assignment, structured label definitions, and output format constraints — none of which were present in the initial prompt.

---

## System Architecture

```
TaskConfig
    │
    ▼
buildMetaPrompt(τ_t)
    │  constructs M(τ_t) from trajectory τ_t sorted ascending by score
    ▼
LLM_optimizer (claude-sonnet-4-6)
    │  generates N candidate prompts
    ▼
[p_1, p_2, ..., p_N]
    │
    ▼ (parallel evaluation)
LLM_evaluator × |E| calls per candidate
    │  f(p_i) = (1/|E|) Σ 1[LLM(p_i, e.input) == e.expected]
    ▼
scores [s_1, s_2, ..., s_N]
    │
    ▼
τ_{t+1} = τ_t ∪ {(best_p, best_s)}
    │
    └──► yield IterationResult → UI streaming layer
              │
              ▼
         PostgreSQL (Neon)
         optimization_runs + prompt_candidates + example_evaluations
```

**Phase 1 — Engine (`src/`):** TypeScript OPRO core. `buildMetaPrompt`, `evaluatePrompt`, `generateCandidates`, `runOPRO` async generator. Zero UI dependencies. Testable in isolation via `test-sentiment.ts` and `test-financial.ts`.

**Phase 2 — Next.js UI (`app/`):** Streaming API route consumes the async generator, forwards IterationResults as NDJSON. React frontend with live trajectory chart (Recharts), prompt diff view, and iteration log. PostgreSQL persistence via Neon.

**Phase 3 — Math Walkthrough:** Interactive KaTeX-rendered explanation of the algorithm with live annotation during active optimization runs — highlighting which mathematical step is currently executing.

---

## Reproducibility and Auditability

Every optimization run produces a complete audit trail stored in PostgreSQL:

- Full trajectory τ_t with all (prompt, score) pairs at every iteration
- The exact meta-prompt M(τ_t) used at each iteration — reproducible by construction
- Per-example evaluation results: input, expected output, actual output, correctness flag
- Run metadata: task description, task type, iteration count, candidates per iteration, timestamp

This means any optimization result can be fully reconstructed. Given the same TaskConfig, the same Anthropic API model version, and the stored meta-prompts, the experiment is reproducible. For regulated financial applications, this audit trail satisfies model governance documentation requirements — the search procedure, not just the result, is logged.

```sql
CREATE TABLE optimization_runs (
    id               UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    task_description TEXT         NOT NULL,
    task_type        TEXT         DEFAULT 'general',
    examples         JSONB        NOT NULL,
    iterations       INT          NOT NULL,
    candidates_per_iter INT       NOT NULL,
    initial_prompt   TEXT,
    best_prompt      TEXT,
    best_score       FLOAT,
    status           TEXT         DEFAULT 'running',
    created_at       TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE prompt_candidates (
    id               UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    run_id           UUID         REFERENCES optimization_runs(id) ON DELETE CASCADE,
    iteration        INT,
    prompt_text      TEXT,
    score            FLOAT,
    meta_prompt_used TEXT,
    created_at       TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE example_evaluations (
    id               UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
    candidate_id     UUID         REFERENCES prompt_candidates(id) ON DELETE CASCADE,
    example_input    TEXT,
    expected_output  TEXT,
    actual_output    TEXT,
    is_correct       BOOLEAN
);
```

---

## Project Structure

```
PromptDescent/
├── src/
│   ├── types.ts                  # TaskConfig, Trajectory, CandidateResult, IterationResult
│   ├── opro-engine.ts            # buildMetaPrompt, evaluatePrompt, generateCandidates, runOPRO
│   ├── test-sentiment.ts         # Movie review benchmark
│   └── test-financial.ts         # Earnings call benchmark
├── app/
│   ├── page.tsx                  # Live Optimizer | How It Works tabs
│   ├── api/
│   │   ├── optimize/route.ts     # Streaming POST → NDJSON
│   │   └── runs/route.ts         # GET/POST run persistence
│   └── components/
│       ├── OptimizerForm.tsx
│       ├── TrajectoryChart.tsx
│       ├── PromptDiff.tsx
│       ├── IterationLog.tsx
│       └── MathWalkthrough.tsx
├── .env.example
├── package.json
└── README.md
```

---

## Setup

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env
# Set ANTHROPIC_API_KEY and DATABASE_URL in .env

# Run benchmark tests
npm run test:sentiment
npm run test:financial

# Start development server (Phase 2+)
npm run dev
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key for optimizer and evaluator LLM calls |
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `NEXT_PUBLIC_APP_URL` | Application URL (default: http://localhost:3000) |

---

## References

1. **Wei et al. 2023** — *Large Language Models as Optimizers*. Google DeepMind. arXiv:2309.03409. Primary paper — source of the OPRO algorithm, meta-prompt construction, and evaluation procedure implemented here.

2. **Snoek, Larochelle & Adams 2012** — *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS 2012. Foundational BO reference — the surrogate/acquisition function framework used in the BO comparison above.

3. **Hansen 2006** — *The CMA Evolution Strategy: A Comparing Review*. In: Towards a New Evolutionary Computation. Springer. Foundational CMA-ES reference — the covariance matrix adaptation framework used in the evolutionary methods comparison.

4. **Guo et al. 2023** — *EvoPrompt: Language Model-Based Evolutionary Prompt Optimization*. arXiv:2309.08532. Related work applying evolutionary algorithms to prompt optimization — a discrete-space alternative to OPRO using genetic operators rather than LLM-as-optimizer.

5. **Nesterov & Spokoiny 2017** — *Random Gradient-Free Minimization of Convex Functions*. Foundations of Computational Mathematics. Foundational zeroth-order optimization reference — the finite-difference gradient estimation framework referenced in the convergence analysis above.

6. **Flaxman, Kalai & McMahan 2005** — *Online Convex Optimization in the Bandit Setting: Gradient Descent without a Gradient*. SODA 2005. Early foundational work on gradient-free online optimization — establishes the theoretical lineage that OPRO's empirical approach connects to.

---

## Implementation Notes

**No prompt optimization libraries.** This is a direct implementation of Wei et al. 2023. No DSPy, no LangChain, no LlamaIndex. Every component — meta-prompt construction, trajectory management, candidate parsing, evaluation scoring — is implemented from the paper.

**AsyncGenerator architecture.** `runOPRO` is an async generator that yields one `IterationResult` per iteration. This enables streaming to the UI via `for-await-of` without buffering the full trajectory, and makes the algorithm independently testable from the UI layer.

**Evaluation independence.** Each (prompt, example) pair is evaluated in a separate API call. This preserves evaluation independence (no cross-contamination between examples in a single context), matches the paper's experimental setup, and enables per-example debugging in the iteration log.

**Zeroth-order optimization analogy.** The OPRO update rule is structurally isomorphic to zeroth-order gradient estimation: the ascending-sorted score trajectory provides implicit finite-difference information about the objective landscape, and the LLM uses this signal to propose candidate prompts in higher-scoring regions of P. This framing connects PromptDescent to a well-studied family of gradient-free optimization methods (Nesterov & Spokoiny 2017, Flaxman et al. 2005) and situates it within the broader optimization literature beyond the prompt engineering community.

---

Built by David Johnson — [davejohnson.io](https://davejohnson.io) | [github.com/datadave22](https://github.com/datadave22) | Production AI Platform Engineer
