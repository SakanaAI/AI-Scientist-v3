import assert from "node:assert/strict";
import test, { after, afterEach, before, beforeEach } from "node:test";
import { fileURLToPath } from "node:url";
import { act, createElement } from "react";
import { JSDOM } from "jsdom";
import { createServer } from "vite";

const dom = new JSDOM("<!doctype html><div id='root'></div>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { createRoot } = await import("react-dom/client");
const originalFetch = globalThis.fetch;
const container = document.getElementById("root");
let server;
let ProgramDiffViewer;
let root;
let requests;
let respond;

const beforePath = "ideas/first/program_0";
const afterPath = "ideas/second/program_0";
const location = (path) => ({ path });
const program = (path) => ({
  label: "Best program",
  attempt_name: null,
  program_info: location(path),
});
const ancestors = {
  idea_id: "second",
  target_program: location(afterPath),
  ancestors: [
    {
      idea_id: "first",
      title: "First",
      cycle_index: 0,
      worker_name: null,
      program_info: location(beforePath),
      programs: [program(beforePath)],
    },
  ],
  sibling_programs: [],
};
const file = (relpath, status = "modified", is_binary = false) => ({
  relpath,
  status,
  is_binary,
  from_path: `${beforePath}/${relpath}`,
  to_path: `${afterPath}/${relpath}`,
  from_exists: status !== "added",
  to_exists: status !== "removed",
  from_size_bytes: status === "added" ? null : 10,
  to_size_bytes: status === "removed" ? null : 20,
});
const directory = (files) => ({
  files,
  stats: {
    added: files.filter((file) => file.status === "added").length,
    removed: files.filter((file) => file.status === "removed").length,
    modified: files.filter((file) => file.status === "modified").length,
    unchanged: 0,
  },
});
const textDiff = (text) => ({
  lines: [{ type: "addition", content: text, from_line: null, to_line: 1 }],
  stats: { additions: 1, deletions: 0, hunks: 1 },
});

before(async () => {
  server = await createServer({
    root: fileURLToPath(new URL("..", import.meta.url)),
    server: { middlewareMode: true, watch: null },
    appType: "custom",
  });
  ({ ProgramDiffViewer } = await server.ssrLoadModule("/src/components/ProgramDiffViewer.tsx"));
});

beforeEach(() => {
  requests = [];
  respond = (url) => {
    if (url.pathname.includes("idea-ancestors")) return ancestors;
    if (url.pathname.endsWith("program-dir-diff")) {
      return directory([
        file("src/main.py"),
        file("src/new.py", "added"),
        file("src/old.py", "removed"),
        file("src/image.bin", "modified", true),
      ]);
    }
    if (url.pathname.endsWith("program-diff")) {
      return textDiff(url.searchParams.get("to_path"));
    }
    throw new Error(`Unexpected request: ${url}`);
  };
  globalThis.fetch = async (input) => {
    const url = new URL(input, "http://localhost");
    requests.push(url);
    return Response.json(await respond(url));
  };
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  globalThis.fetch = originalFetch;
});

after(async () => {
  await server.close();
  dom.window.close();
});

async function render(props = {}) {
  await act(async () => {
    root.render(createElement(ProgramDiffViewer, {
      runId: "task/codex/run",
      ideaId: "second",
      ideaTitle: "Second",
      ...props,
    }));
  });
}

function fileSelector() {
  return container.querySelector('select[aria-label="File to compare"]');
}

async function selectFile(path) {
  const select = fileSelector();
  const option = [...select.options].find((item) => item.textContent.includes(path));
  assert.ok(option, `File selector contains ${path}`);

  await act(async () => {
    select.value = option.value;
    select.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}

test("opens the comparison and loads changed files before requesting a file diff", async () => {
  await render();

  assert.match(container.textContent, /Program Diff/);
  const directoryRequest = requests.find((url) => url.pathname.endsWith("program-dir-diff"));
  assert.equal(directoryRequest?.searchParams.get("from_dir"), beforePath);
  assert.equal(directoryRequest?.searchParams.get("to_dir"), afterPath);
  assert.match(container.querySelector(".diff-content").textContent, /src\/main\.py/);
  assert.equal(fileSelector().options.length, 4);
});

test("selecting added and removed files loads their own comparisons", async () => {
  await render();

  for (const path of ["src/new.py", "src/old.py"]) {
    await selectFile(path);
    const request = requests.at(-1);
    assert.equal(request.searchParams.get("from_path"), `${beforePath}/${path}`);
    assert.equal(request.searchParams.get("to_path"), `${afterPath}/${path}`);
    assert.ok(container.querySelector(".diff-content").textContent.includes(path));
  }
});

test("binary files show a clear message without requesting a text diff", async () => {
  await render();
  const count = requests.length;
  await selectFile("src/image.bin");

  assert.equal(requests.length, count);
  assert.match(container.querySelector(".diff-content").textContent, /binary/i);
  assert.equal(container.querySelector(".diff-table"), null);
});

test("identical programs show one empty result instead of loading forever", async () => {
  const fallback = respond;
  respond = (url) => url.pathname.endsWith("program-dir-diff") ? directory([]) : fallback(url);
  await render();

  assert.equal(container.querySelectorAll(".diff-empty").length, 1);
  assert.match(container.textContent, /No differences found/);
  assert.equal(container.querySelector(".diff-loading"), null);
  assert.ok(!requests.some((url) => url.pathname.endsWith("program-diff")));
});

test("a late file response cannot replace the newly selected file", async () => {
  let resolveOld;
  const fallback = respond;
  respond = (url) => {
    if (url.pathname.endsWith("program-diff") && url.searchParams.get("to_path").endsWith("main.py")) {
      return new Promise((resolve) => {
        resolveOld = resolve;
      });
    }
    return fallback(url);
  };

  await render();
  await selectFile("src/new.py");
  await act(async () => resolveOld(textDiff("STALE RESPONSE")));

  assert.match(container.querySelector(".diff-content").textContent, /src\/new\.py/);
  assert.doesNotMatch(container.textContent, /STALE RESPONSE/);
});
