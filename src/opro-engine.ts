import Anthropic from "@anthropic-ai/sdk";
import type { TaskConfig, Trajectory, CandidateResult, IterationResult } from "./types.js";

const MODEL = "claude-sonnet-4-6";

/**
 * Constructs M(τ_t) from trajectory τ_t.
 * Implements Section 3 of Wei et al. 2023: meta-prompt construction.
 * Trajectory is sorted ascending by score before formatting — the ascending
 * order exposes the direction of improvement to LLM_optimizer, analogous
 * to providing sign(∂f/∂p) in gradient descent.
 */
export function buildMetaPrompt(task: TaskConfig, trajectory: Trajectory): string {
  const sorted = [...trajectory].sort((a, b) => a.score - b.score);

  const parts: string[] = [];

  parts.push(`TASK: ${task.task_description}\n`);

  parts.push("PREVIOUS ATTEMPTS (sorted by score, ascending):");
  for (const entry of sorted) {
    parts.push(`Score: ${entry.score.toFixed(2)}\nPrompt: ${entry.prompt}\n---`);
  }

  parts.push("\nEXAMPLES:");
  const exampleSample = task.examples.slice(0, 3);
  for (const ex of exampleSample) {
    parts.push(`Input: ${ex.input}\nExpected: ${ex.expected_output}\n`);
  }

  parts.push(
    `Generate ${task.candidates_per_iteration} new prompts that score higher than all previous attempts.\n` +
    `Output ONLY a numbered list:\n` +
    Array.from({ length: task.candidates_per_iteration }, (_, i) => `${i + 1}. [prompt text]`).join("\n")
  );

  return parts.join("\n");
}

/**
 * Implements f(p) = (1/|E|) Σ_{e∈E} 1[LLM(p, e.input) == e.expected].
 * One API call per example preserves evaluation independence — no cross-
 * contamination between examples in a single context window.
 * Paper reference: Wei et al. 2023, Section 3, evaluation procedure.
 */
export async function evaluatePrompt(
  prompt: string,
  examples: TaskConfig["examples"],
  client: Anthropic
): Promise<{ score: number; evaluations: CandidateResult["evaluations"] }> {
  const evaluations: CandidateResult["evaluations"] = [];

  for (const example of examples) {
    try {
      const response = await client.messages.create({
        model: MODEL,
        max_tokens: 20,
        system: "You are an evaluator. Apply the given prompt to the input and output ONLY the answer, nothing else.",
        messages: [
          {
            role: "user",
            content: `Prompt: ${prompt}\n\nInput: ${example.input}\n\nOutput:`,
          },
        ],
      });

      const actual = response.content
        .filter((b): b is Anthropic.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("")
        .trim()
        .toLowerCase();

      const expected = example.expected_output.trim().toLowerCase();
      const correct = actual === expected;

      evaluations.push({
        input: example.input,
        expected: example.expected_output,
        actual,
        correct,
      });
    } catch (err) {
      console.error(`  [evaluatePrompt] API error on example "${example.input.slice(0, 40)}...":`, err);
      evaluations.push({
        input: example.input,
        expected: example.expected_output,
        actual: "",
        correct: false,
      });
    }
  }

  const score = evaluations.filter((e) => e.correct).length / evaluations.length;
  return { score, evaluations };
}

/**
 * Parses N candidate prompts out of the optimizer LLM's raw text response.
 * Pure function — no network access — so it's unit-testable in isolation
 * from the API call in generateCandidates below.
 */
export function parseCandidates(raw: string, n: number, fallback: string): string[] {
  // Primary: parse numbered list "1. ..." "2. ..." etc.
  const numbered = raw.split(/\n(?=\d+\.\s)/).map((line) =>
    line.replace(/^\d+\.\s*/, "").trim()
  ).filter((line) => line.length > 0);

  if (numbered.length >= n) {
    return numbered.slice(0, n);
  }

  // Fallback: split on double newline
  const byParagraph = raw.split(/\n\n+/).map((s) => s.trim()).filter((s) => s.length > 0);
  if (byParagraph.length >= 1) {
    return byParagraph.slice(0, n);
  }

  // Last resort: return truncated fallback text as single candidate
  console.warn("  [parseCandidates] Parsing failed; using fallback candidate.");
  return [fallback.slice(0, 200)];
}

/**
 * Generates N candidate prompts from LLM_optimizer conditioned on M(τ_t).
 * One API call returns a numbered list; parsed with regex fallback.
 * max_tokens=800 accommodates up to ~5 candidates of reasonable length.
 */
export async function generateCandidates(
  metaPrompt: string,
  n: number,
  client: Anthropic
): Promise<string[]> {
  try {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 800,
      system: "You are a prompt optimization assistant. Generate improved prompt candidates based on the performance history provided.",
      messages: [{ role: "user", content: metaPrompt }],
    });

    const raw = response.content
      .filter((b): b is Anthropic.TextBlock => b.type === "text")
      .map((b) => b.text)
      .join("");

    return parseCandidates(raw, n, metaPrompt);
  } catch (err) {
    console.error("  [generateCandidates] API error:", err);
    return [metaPrompt.slice(0, 200)];
  }
}

/**
 * Main OPRO loop — Algorithm 1 from Wei et al. 2023.
 * AsyncGenerator yields one IterationResult per iteration t.
 * Each yield: p_{t+1} ~ LLM(M(τ_t)), scored via f(p), trajectory updated.
 *
 * Complexity: O(T * N * |E|) API calls total where T=iterations, N=candidates,
 * |E|=examples. Candidate evaluation is sequential per-candidate; could be
 * parallelized with Promise.all at the cost of rate-limit headroom.
 */
export async function* runOPRO(
  task: TaskConfig,
  client: Anthropic
): AsyncGenerator<IterationResult> {
  // Initialize trajectory with the initial prompt
  console.log("\nEvaluating initial prompt...");
  const { score: initialScore, evaluations: initialEvals } = await evaluatePrompt(
    task.initial_prompt,
    task.examples,
    client
  );
  console.log(`  Initial prompt score: ${(initialScore * 100).toFixed(1)}%`);

  const trajectory = [{ prompt: task.initial_prompt, score: initialScore }];

  // Iterate t = 1..T
  for (let iteration = 1; iteration <= task.iterations; iteration++) {
    console.log(`\n${"═".repeat(60)}`);
    console.log(`Iteration ${iteration} / ${task.iterations}`);
    console.log(`${"═".repeat(60)}`);

    // Build M(τ_t)
    const metaPrompt = buildMetaPrompt(task, trajectory);

    // Generate N candidates: {p^(1)_{t+1}, ..., p^(N)_{t+1}} ~ LLM(M(τ_t))
    console.log(`  Generating ${task.candidates_per_iteration} candidates...`);
    const candidateTexts = await generateCandidates(
      metaPrompt,
      task.candidates_per_iteration,
      client
    );

    // Evaluate each candidate: s^(i) = f(p^(i))
    const candidates: CandidateResult[] = [];
    for (let i = 0; i < candidateTexts.length; i++) {
      const promptText = candidateTexts[i];
      console.log(`\n  Candidate ${i + 1}: "${promptText.slice(0, 80)}${promptText.length > 80 ? "..." : ""}"`);

      let score = 0;
      let evaluations: CandidateResult["evaluations"] = [];

      try {
        ({ score, evaluations } = await evaluatePrompt(promptText, task.examples, client));
      } catch (err) {
        console.error(`  Error evaluating candidate ${i + 1}:`, err);
      }

      console.log(`  Score: ${(score * 100).toFixed(1)}%`);
      candidates.push({ prompt_text: promptText, score, evaluations });
    }

    // Find best candidate from this iteration: p*_{t+1} = argmax_i f(p^(i))
    const bestCandidate = candidates.reduce((best, c) =>
      c.score > best.score ? c : best
    );

    // Trajectory update: τ_{t+1} = τ_t ∪ {(p*_{t+1}, s*_{t+1})}
    trajectory.push({ prompt: bestCandidate.prompt_text, score: bestCandidate.score });

    const overallBest = trajectory.reduce((best, e) => (e.score > best.score ? e : best));

    console.log(`\n  Best this iteration: ${(bestCandidate.score * 100).toFixed(1)}%`);
    console.log(`  Overall best so far: ${(overallBest.score * 100).toFixed(1)}%`);

    yield {
      iteration,
      candidates,
      best_prompt: overallBest.prompt,
      best_score: overallBest.score,
      trajectory: [...trajectory],
    };
  }
}
