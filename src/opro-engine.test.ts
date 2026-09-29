import { test } from "node:test";
import assert from "node:assert/strict";
import { buildMetaPrompt, parseCandidates } from "./opro-engine.js";
import type { TaskConfig, Trajectory } from "./types.js";

const baseTask: TaskConfig = {
  task_description: "Classify sentiment as positive or negative.",
  task_type: "general",
  initial_prompt: "Classify the sentiment.",
  iterations: 3,
  candidates_per_iteration: 2,
  examples: [
    { input: "Great movie!", expected_output: "positive" },
    { input: "Terrible movie.", expected_output: "negative" },
    { input: "Loved it.", expected_output: "positive" },
    { input: "Hated it.", expected_output: "negative" },
  ],
};

test("buildMetaPrompt sorts trajectory ascending by score regardless of input order", () => {
  const trajectory: Trajectory = [
    { prompt: "high scorer", score: 0.9 },
    { prompt: "low scorer", score: 0.1 },
    { prompt: "mid scorer", score: 0.5 },
  ];

  const metaPrompt = buildMetaPrompt(baseTask, trajectory);
  const lowIdx = metaPrompt.indexOf("low scorer");
  const midIdx = metaPrompt.indexOf("mid scorer");
  const highIdx = metaPrompt.indexOf("high scorer");

  assert.ok(lowIdx < midIdx && midIdx < highIdx, "expected ascending score order in meta-prompt");
});

test("buildMetaPrompt includes task description and only the first 3 examples", () => {
  const metaPrompt = buildMetaPrompt(baseTask, [{ prompt: "p", score: 0.5 }]);

  assert.ok(metaPrompt.includes(baseTask.task_description));
  assert.ok(metaPrompt.includes("Great movie!"));
  assert.ok(metaPrompt.includes("Terrible movie."));
  assert.ok(metaPrompt.includes("Loved it."));
  assert.ok(!metaPrompt.includes("Hated it."), "expected only the first 3 examples to be included");
});

test("buildMetaPrompt requests exactly candidates_per_iteration numbered slots", () => {
  const metaPrompt = buildMetaPrompt(baseTask, [{ prompt: "p", score: 0.5 }]);

  assert.ok(metaPrompt.includes("1. [prompt text]"));
  assert.ok(metaPrompt.includes("2. [prompt text]"));
  assert.ok(!metaPrompt.includes("3. [prompt text]"));
});

test("parseCandidates parses a well-formed numbered list", () => {
  const raw = "1. Classify this review as positive or negative.\n2. Determine the sentiment: positive or negative.";
  const result = parseCandidates(raw, 2, "fallback");

  assert.deepEqual(result, [
    "Classify this review as positive or negative.",
    "Determine the sentiment: positive or negative.",
  ]);
});

test("parseCandidates truncates when the model returns more candidates than requested", () => {
  const raw = "1. First.\n2. Second.\n3. Third.";
  const result = parseCandidates(raw, 2, "fallback");

  assert.deepEqual(result, ["First.", "Second."]);
});

test("parseCandidates falls back to paragraph splitting when numbering is absent", () => {
  const raw = "First candidate prompt text.\n\nSecond candidate prompt text.";
  const result = parseCandidates(raw, 2, "fallback");

  assert.deepEqual(result, ["First candidate prompt text.", "Second candidate prompt text."]);
});

test("parseCandidates uses the truncated fallback when nothing parseable is found", () => {
  const result = parseCandidates("", 2, "some long fallback text");
  assert.deepEqual(result, ["some long fallback text"]);
});
