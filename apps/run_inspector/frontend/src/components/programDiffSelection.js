export function defaultDiffFile(files) {
  const textFiles = files.filter((file) => !file.is_binary);
  return textFiles.find((file) => file.status === "modified") ?? textFiles[0] ?? files[0] ?? null;
}

export function emptyDiffMessage(file) {
  if (file?.status === "added") return "Empty file added.";
  if (file?.status === "removed") return "Empty file removed.";
  return "No differences found.";
}
