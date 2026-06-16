export function formatCheckpoint(path) {
  if (!path || path === "none") return "None (Scratch)";
  
  if (path === "pretrained.pth") return "Pretrained Base";
  
  // If it's a root checkpoint
  if (path === "checkpoint_best.pth" || path === "checkpoint_last.pth") {
    return "Default Run";
  }
  
  // If it's in a subdirectory like sweep_1/checkpoint_best.pth -> sweep_1
  if (path.includes("/")) {
    const parts = path.split("/");
    return parts[parts.length - 2]; // Return just the run name (e.g. sweep_1)
  }
  
  return path;
}

export function formatNumber(num, decimals = 2) {
  if (num === null || num === undefined) return "0";
  return Number(num).toFixed(decimals);
}

export function formatInteger(num) {
  if (num === null || num === undefined) return "0";
  return Number(num).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
}
