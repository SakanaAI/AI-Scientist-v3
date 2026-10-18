import type { ProgramDirDiffFile } from "../types";

export function defaultDiffFile(
  files: readonly ProgramDirDiffFile[],
): ProgramDirDiffFile | null;

export function emptyDiffMessage(file: ProgramDirDiffFile | null): string;
