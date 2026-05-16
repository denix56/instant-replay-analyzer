import { spawn } from "node:child_process";
import { delimiter, join } from "node:path";
import { homedir } from "node:os";

const mode = process.argv[2] || "dev";
const cargoBin = join(homedir(), ".cargo", "bin");
const bin = join(process.cwd(), "node_modules", ".bin", process.platform === "win32" ? "tauri.cmd" : "tauri");
const env = {
  ...process.env,
  PATH: [cargoBin, process.env.PATH || ""].filter(Boolean).join(delimiter),
};

const child = spawn(bin, [mode], {
  env,
  stdio: "inherit",
  shell: process.platform === "win32",
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 1);
});

child.on("error", (error) => {
  console.error(`Failed to start Tauri CLI: ${error.message}`);
  process.exit(1);
});
