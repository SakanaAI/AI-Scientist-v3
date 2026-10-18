import assert from "node:assert/strict";
import test from "node:test";
import { defaultDiffFile, emptyDiffMessage } from "../src/components/programDiffSelection.js";

const file = (relpath, status, is_binary = false) => ({ relpath, status, is_binary });

test("default file prefers a modified text file over added files", () => {
  const files = [file("src/helper.py", "added"), file("src/program.py", "modified")];

  assert.equal(defaultDiffFile(files).relpath, "src/program.py");
});

test("default file skips binary files", () => {
  const files = [file("src/weights.bin", "modified", true), file("src/helper.py", "added")];

  assert.equal(defaultDiffFile(files).relpath, "src/helper.py");
});

test("default file falls back to a binary file when no text file exists", () => {
  const files = [file("src/weights.bin", "modified", true)];

  assert.equal(defaultDiffFile(files).relpath, "src/weights.bin");
});

test("default file is null for an empty comparison", () => {
  assert.equal(defaultDiffFile([]), null);
});

test("empty diff message names added and removed empty files", () => {
  assert.equal(emptyDiffMessage(file("src/a.py", "added")), "Empty file added.");
  assert.equal(emptyDiffMessage(file("src/a.py", "removed")), "Empty file removed.");
  assert.equal(emptyDiffMessage(null), "No differences found.");
});
