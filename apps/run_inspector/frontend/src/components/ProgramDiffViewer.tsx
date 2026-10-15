import { useEffect, useState } from "react";
import { fetchIdeaAncestors, fetchProgramDiff, fetchProgramDirDiff } from "../api/client";
import type {
  AncestorEntry,
  ProgramDiff,
  ProgramDirDiff,
  ProgramDirDiffFile,
  ProgramLocation,
  SiblingProgramEntry,
} from "../types";

interface ProgramDiffViewerProps {
  runId: string;
  ideaId: string;
  ideaTitle: string;
  workerName?: string | null;
  ideaPath?: string | null;
}

/** A unified selection: either the current target, an ancestor, or a sibling program. */
type CompareTarget =
  | { kind: "target"; entry: { program_info: ProgramLocation } }
  | { kind: "ancestor"; index: number; entry: AncestorEntry }
  | { kind: "sibling"; index: number; entry: SiblingProgramEntry };

function targetKey(target: CompareTarget): string {
  if (target.kind === "target") return "target";
  return `${target.kind}:${target.index}`;
}

/** Get available programs for a given selection. */
function getEntryPrograms(
  target: CompareTarget,
  targetProgram: ProgramLocation | null,
  siblings: SiblingProgramEntry[],
): SiblingProgramEntry[] {
  if (target.kind === "target") {
    const progs: SiblingProgramEntry[] = [];
    if (targetProgram) {
      progs.push({ label: "Best program", attempt_name: null, program_info: targetProgram });
    }
    progs.push(...siblings);
    return progs;
  }
  if (target.kind === "ancestor") {
    return target.entry.programs || [];
  }
  if (target.kind === "sibling") {
    if (target.entry.program_info) {
      return [target.entry];
    }
    return [];
  }
  return [];
}

/** Renders <option>/<optgroup> elements for a cycle selector dropdown. */
function CycleOptions({
  targetProgram,
  ancestors,
  siblings,
  ideaTitle,
}: {
  targetProgram: ProgramLocation | null;
  ancestors: AncestorEntry[];
  siblings: SiblingProgramEntry[];
  ideaTitle: string;
}) {
  const hasAncestors = ancestors.length > 0;
  const hasSiblings = siblings.length > 0;
  return (
    <>
      {targetProgram && (
        <optgroup label="Current idea (best program)">
          <option value="target">
            {ideaTitle} (best)
          </option>
        </optgroup>
      )}
      {hasAncestors && (
        <optgroup label="Previous cycles (ancestors)">
          {ancestors.map((anc, idx) => (
            <option key={`anc-${anc.idea_id}`} value={`ancestor:${idx}`} disabled={!anc.program_info}>
              Cycle {anc.cycle_index} &mdash; {anc.title}
              {!anc.program_info ? " (no program)" : ""}
            </option>
          ))}
        </optgroup>
      )}
      {hasSiblings && (
        <optgroup label="Current cycle (attempts)">
          {siblings.map((sib, idx) => (
            <option key={`sib-${idx}`} value={`sibling:${idx}`} disabled={!sib.program_info}>
              {sib.label}
              {!sib.program_info ? " (no program)" : ""}
            </option>
          ))}
        </optgroup>
      )}
    </>
  );
}

function parseTargetKey(
  val: string,
  targetProgram: ProgramLocation | null,
  ancestors: AncestorEntry[],
  siblings: SiblingProgramEntry[],
): CompareTarget | null {
  if (val === "target" && targetProgram) {
    return { kind: "target", entry: { program_info: targetProgram } };
  }
  const [kind, idxStr] = val.split(":");
  const idx = parseInt(idxStr, 10);
  if (kind === "ancestor" && ancestors[idx]) {
    return { kind: "ancestor", index: idx, entry: ancestors[idx] };
  }
  if (kind === "sibling" && siblings[idx]) {
    return { kind: "sibling", index: idx, entry: siblings[idx] };
  }
  return null;
}

export function ProgramDiffViewer({ runId, ideaId, ideaTitle, workerName }: ProgramDiffViewerProps) {
  const [ancestors, setAncestors] = useState<AncestorEntry[]>([]);
  const [siblings, setSiblings] = useState<SiblingProgramEntry[]>([]);
  const [targetProgram, setTargetProgram] = useState<ProgramLocation | null>(null);
  /** The "from" (left / old) side of the diff */
  const [fromSelected, setFromSelected] = useState<CompareTarget | null>(null);
  /** The "to" (right / new) side of the diff — defaults to target program */
  const [toSelected, setToSelected] = useState<CompareTarget | null>(null);
  /** Programs available for the from/to selections */
  const [fromPrograms, setFromPrograms] = useState<SiblingProgramEntry[]>([]);
  const [toPrograms, setToPrograms] = useState<SiblingProgramEntry[]>([]);
  /** Selected program indices within fromPrograms/toPrograms */
  const [fromProgIdx, setFromProgIdx] = useState<number>(0);
  const [toProgIdx, setToProgIdx] = useState<number>(0);

  const [diff, setDiff] = useState<ProgramDiff | null>(null);
  const [dirDiff, setDirDiff] = useState<ProgramDirDiff | null>(null);
  const [selectedFile, setSelectedFile] = useState<ProgramDirDiffFile | null>(null);
  const [loadingDiff, setLoadingDiff] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setAncestors([]);
    setSiblings([]);
    setTargetProgram(null);
    setFromSelected(null);
    setToSelected(null);
    setFromPrograms([]);
    setToPrograms([]);
    setFromProgIdx(0);
    setToProgIdx(0);
    setDiff(null);
    fetchIdeaAncestors(runId, ideaId, workerName)
      .then((data) => {
        if (cancelled) return;
        const fetchedAncestors = data.ancestors || [];
        const fetchedSiblings = data.sibling_programs || [];
        setAncestors(fetchedAncestors);
        setSiblings(fetchedSiblings);
        setTargetProgram(data.target_program);
        // Auto-select "to" = target program (current idea's best)
        if (data.target_program) {
          const toTarget: CompareTarget = { kind: "target", entry: { program_info: data.target_program } };
          setToSelected(toTarget);
          const toProgs = getEntryPrograms(toTarget, data.target_program, fetchedSiblings);
          setToPrograms(toProgs);
          setToProgIdx(0);
        }
        // Auto-select "from" = first ancestor (immediate parent) if available
        if (fetchedAncestors.length > 0 && fetchedAncestors[0].program_info) {
          const fromTarget: CompareTarget = { kind: "ancestor", index: 0, entry: fetchedAncestors[0] };
          setFromSelected(fromTarget);
          const fromProgs = getEntryPrograms(fromTarget, data.target_program, fetchedSiblings);
          setFromPrograms(fromProgs);
          // Default to best program (first with matching path, or index 0)
          const bestIdx = fromProgs.findIndex(
            (p) => p.program_info?.path === fetchedAncestors[0].program_info?.path,
          );
          setFromProgIdx(bestIdx >= 0 ? bestIdx : 0);
        } else if (fetchedSiblings.length > 0 && fetchedSiblings[0].program_info) {
          const fromTarget: CompareTarget = { kind: "sibling", index: 0, entry: fetchedSiblings[0] };
          setFromSelected(fromTarget);
          const fromProgs = getEntryPrograms(fromTarget, data.target_program, fetchedSiblings);
          setFromPrograms(fromProgs);
          setFromProgIdx(0);
        }
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, ideaId, workerName]);

  // Program locations are directories. List their changed files before loading a file diff.
  useEffect(() => {
    setDirDiff(null);
    setSelectedFile(null);
    setDiff(null);
    if (!fromSelected || !toSelected) {
      setLoadingDiff(false);
      return;
    }
    const fromPath = fromPrograms[fromProgIdx]?.program_info?.path ?? null;
    const toPath = toPrograms[toProgIdx]?.program_info?.path ?? null;
    if (!fromPath || !toPath) {
      setLoadingDiff(false);
      return;
    }

    let cancelled = false;
    const loadDirectoryDiff = async () => {
      setError(null);
      setLoadingDiff(true);
      try {
        const payload = await fetchProgramDirDiff(runId, fromPath, toPath);
        if (cancelled) return;
        setDirDiff(payload);
        const firstFile = payload.files.find((file) => !file.is_binary) ?? payload.files[0] ?? null;
        setSelectedFile(firstFile);
        if (!firstFile || firstFile.is_binary) {
          setLoadingDiff(false);
        }
      } catch (error) {
        if (cancelled) return;
        setError(error instanceof Error ? error.message : String(error));
        setLoadingDiff(false);
      }
    };

    void loadDirectoryDiff();
    return () => {
      cancelled = true;
    };
  }, [runId, fromSelected, toSelected, fromProgIdx, toProgIdx, fromPrograms, toPrograms]);

  useEffect(() => {
    setDiff(null);
    if (!selectedFile) return;
    setError(null);
    if (selectedFile.is_binary) {
      setLoadingDiff(false);
      return;
    }

    const { from_path: fromPath, to_path: toPath } = selectedFile;
    let cancelled = false;
    const loadDiff = async () => {
      setLoadingDiff(true);
      try {
        const payload = await fetchProgramDiff(runId, fromPath, toPath);
        if (cancelled) return;
        setDiff(payload);
      } catch (error) {
        if (cancelled) return;
        setError(error instanceof Error ? error.message : String(error));
      } finally {
        if (!cancelled) {
          setLoadingDiff(false);
        }
      }
    };

    void loadDiff();
    return () => {
      cancelled = true;
    };
  }, [runId, selectedFile]);

  const handleFileChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedFile(dirDiff?.files[parseInt(e.target.value, 10)] ?? null);
  };

  const handleFromChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const parsed = parseTargetKey(e.target.value, targetProgram, ancestors, siblings);
    if (parsed) {
      setFromSelected(parsed);
      const progs = getEntryPrograms(parsed, targetProgram, siblings);
      setFromPrograms(progs);
      // Default to best program if available
      if (parsed.kind === "ancestor" && parsed.entry.program_info) {
        const bestIdx = progs.findIndex(
          (p) => p.program_info?.path === parsed.entry.program_info?.path,
        );
        setFromProgIdx(bestIdx >= 0 ? bestIdx : 0);
      } else {
        setFromProgIdx(0);
      }
    }
  };

  const handleToChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const parsed = parseTargetKey(e.target.value, targetProgram, ancestors, siblings);
    if (parsed) {
      setToSelected(parsed);
      const progs = getEntryPrograms(parsed, targetProgram, siblings);
      setToPrograms(progs);
      if (parsed.kind === "ancestor" && parsed.entry.program_info) {
        const bestIdx = progs.findIndex(
          (p) => p.program_info?.path === parsed.entry.program_info?.path,
        );
        setToProgIdx(bestIdx >= 0 ? bestIdx : 0);
      } else {
        setToProgIdx(0);
      }
    }
  };

  const handleSwap = () => {
    const prevFrom = fromSelected;
    const prevFromProgs = fromPrograms;
    const prevFromIdx = fromProgIdx;
    setFromSelected(toSelected);
    setFromPrograms(toPrograms);
    setFromProgIdx(toProgIdx);
    setToSelected(prevFrom);
    setToPrograms(prevFromProgs);
    setToProgIdx(prevFromIdx);
  };

  const hasOptions = ancestors.length > 0 || siblings.length > 0 || targetProgram != null;

  return (
    <div className="diff-inline">
      <div className="diff-header">
        <div className="diff-header-left">
          <h3>Program Diff</h3>
          <span className="diff-idea-title">{ideaTitle}</span>
        </div>
      </div>

      {/* Selectors */}
      <div className="diff-selector">
        {loading ? (
          <span className="diff-loading-text">Loading ancestors...</span>
        ) : hasOptions ? (
          <div className="diff-selector-row">
            <div className="diff-selector-field">
              <label>From (old):</label>
              <select
                value={fromSelected ? targetKey(fromSelected) : ""}
                onChange={handleFromChange}
              >
                <CycleOptions
                  targetProgram={targetProgram}
                  ancestors={ancestors}
                  siblings={siblings}
                  ideaTitle={ideaTitle}
                />
              </select>
              {fromPrograms.length > 1 && (
                <>
                  <label>Program:</label>
                  <select
                    value={fromProgIdx}
                    onChange={(e) => setFromProgIdx(parseInt(e.target.value, 10))}
                  >
                    {fromPrograms.map((p, i) => (
                      <option key={i} value={i} disabled={!p.program_info}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </>
              )}
            </div>
            <button
              className="diff-swap-btn"
              onClick={handleSwap}
              title="Swap from/to"
              disabled={!fromSelected || !toSelected}
            >
              &#x21C4;
            </button>
            <div className="diff-selector-field">
              <label>To (new):</label>
              <select
                value={toSelected ? targetKey(toSelected) : ""}
                onChange={handleToChange}
              >
                <CycleOptions
                  targetProgram={targetProgram}
                  ancestors={ancestors}
                  siblings={siblings}
                  ideaTitle={ideaTitle}
                />
              </select>
              {toPrograms.length > 1 && (
                <>
                  <label>Program:</label>
                  <select
                    value={toProgIdx}
                    onChange={(e) => setToProgIdx(parseInt(e.target.value, 10))}
                  >
                    {toPrograms.map((p, i) => (
                      <option key={i} value={i} disabled={!p.program_info}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                </>
              )}
            </div>
          </div>
        ) : (
          <span className="diff-loading-text">No programs available for comparison</span>
        )}
      </div>

      {dirDiff && dirDiff.files.length > 0 ? (
        <div className="diff-selector">
          <label>File:</label>
          <select
            aria-label="File to compare"
            value={selectedFile ? dirDiff.files.indexOf(selectedFile) : ""}
            onChange={handleFileChange}
          >
            {dirDiff.files.map((file, idx) => (
              <option key={file.relpath} value={idx}>
                {file.status.toUpperCase().padEnd(8, " ")} {file.relpath}
                {file.is_binary ? " (binary)" : ""}
              </option>
            ))}
          </select>
          <span className="diff-loading-text">
            {dirDiff.stats.modified} modified, {dirDiff.stats.added} added, {dirDiff.stats.removed} removed
          </span>
        </div>
      ) : null}

      {diff && diff.stats && (
        <div className="diff-stats">
          <span className="diff-stat addition">+{diff.stats.additions}</span>
          <span className="diff-stat removal">&minus;{diff.stats.deletions}</span>
          <span className="diff-stat hunk">
            {diff.stats.hunks} hunk{diff.stats.hunks !== 1 ? "s" : ""}
          </span>
        </div>
      )}

      {error && <div className="diff-error">Error: {error}</div>}

      <div className="diff-content">
        {loadingDiff && !error && (
          <div className="diff-loading">Loading diff...</div>
        )}
        {!loadingDiff && !error && selectedFile?.is_binary && (
          <div className="diff-empty">Binary file: text comparison is not available.</div>
        )}
        {!loadingDiff && !error && (dirDiff?.files.length === 0 || diff?.lines.length === 0) && (
          <div className="diff-empty">No differences found.</div>
        )}
        {diff && diff.lines.length > 0 && (
          <table className="diff-table">
            <tbody>
              {diff.lines.map((line, i) => (
                <tr key={i} className={`diff-line diff-line-${line.type}`}>
                  <td className="diff-line-num left">
                    {line.from_line != null ? line.from_line : ""}
                  </td>
                  <td className="diff-line-num right">
                    {line.to_line != null ? line.to_line : ""}
                  </td>
                  <td className="diff-line-content">
                    {line.type === "hunk_header" ? (
                      <span className="diff-hunk-text">{line.content}</span>
                    ) : (
                      <span>
                        {line.type === "addition" && "+"}
                        {line.type === "removal" && "-"}
                        {line.type === "context" && " "}
                        {line.content}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
