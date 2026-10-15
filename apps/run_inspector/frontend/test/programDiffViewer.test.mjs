import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

test("program comparison renders before any files are loaded", async () => {
  const server = await createServer({
    server: { middlewareMode: true, watch: null },
    optimizeDeps: { noDiscovery: true },
    appType: "custom",
  });
  try {
    const { ProgramDiffViewer } = await server.ssrLoadModule("/src/components/ProgramDiffViewer.tsx");
    const html = renderToStaticMarkup(createElement(ProgramDiffViewer, {
      runId: "task/run",
      ideaId: "idea",
      ideaTitle: "Example",
    }));

    assert.match(html, /Program Diff/);
    assert.match(html, /Loading ancestors/);
  } finally {
    await server.close();
  }
});
