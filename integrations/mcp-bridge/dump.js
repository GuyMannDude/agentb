// Developer Dump — bridge-level JSONL capture of every MCP tool call.
// Phase 1 of the Mnemo v4 observability roadmap. Catches the silent
// tool-failure class (Peter Widget outage) and provides raw material
// for trajectory analysis in Dreaming. Default OFF; flip on with
// MNEMO_DUMP=on. See brain/mnemo-v4-phase1-dump-spec.md for the full
// design rationale.

import {
  existsSync,
  mkdirSync,
  appendFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

export const DUMP_SCHEMA_VERSION = 1;
export const DUMP_PKG_VERSION = "2.9.0";

function resolveConfig(env = process.env) {
  const home = env.HOME || homedir();
  const mode = String(env.MNEMO_DUMP || "off").toLowerCase();
  const dir = env.MNEMO_DUMP_DIR || join(home, ".mnemo-cortex/dumps");
  const retention = parseInt(env.MNEMO_DUMP_RETENTION_DAYS || "0", 10);
  return {
    enabled: mode === "on",
    dir,
    retentionDays: Number.isFinite(retention) ? retention : 0,
  };
}

export class DumpWriter {
  constructor(agentId, config = resolveConfig()) {
    this.agentId = agentId;
    this.enabled = config.enabled;
    this.dir = config.dir;
    this.retentionDays = config.retentionDays;
    this._currentDate = null;
    this._currentPath = null;
    this._failedOnce = false;
  }

  _today() {
    return new Date().toISOString().slice(0, 10);
  }

  _pathFor(date) {
    return join(this.dir, this.agentId, `${date}.jsonl`);
  }

  _ensureFile(today) {
    const path = this._pathFor(today);
    if (this._currentDate === today && existsSync(path)) {
      return path;
    }
    if (existsSync(path)) {
      this._currentDate = today;
      this._currentPath = path;
      return path;
    }
    mkdirSync(dirname(path), { recursive: true });
    const header = {
      ts: new Date().toISOString(),
      kind: "header",
      schema_version: DUMP_SCHEMA_VERSION,
      mnemo_version: DUMP_PKG_VERSION,
      agent_id: this.agentId,
    };
    appendFileSync(path, JSON.stringify(header) + "\n");
    this._currentDate = today;
    this._currentPath = path;
    return path;
  }

  write(event) {
    if (!this.enabled) return;
    try {
      const path = this._ensureFile(this._today());
      const line = { schema_version: DUMP_SCHEMA_VERSION, ...event };
      appendFileSync(path, JSON.stringify(line) + "\n");
      this._failedOnce = false;
    } catch (err) {
      // Fail loud, but never break the bridge — log once per failure
      // streak so a misconfigured dir doesn't flood stderr.
      if (!this._failedOnce) {
        process.stderr.write(
          `[MNEMO DUMP FAIL] Cannot write to ${this.dir}: ${err.message}. ` +
          `Bridge keeps running; tool calls still work. Fix the path or set MNEMO_DUMP=off.\n`
        );
        this._failedOnce = true;
      }
    }
  }

  // Wrap an MCP tool handler so every invocation is logged. When the
  // dump is off, returns the handler untouched (zero overhead).
  // Captures: tool name, params, response, latency, ok/error.
  // Handlers that catch internally and return {isError:true} are still
  // recorded as ok:false — that's the failure class we care about.
  wrap(toolName, handler) {
    if (!this.enabled) return handler;
    const writer = this;
    return async function dumpedHandler(...args) {
      const start = Date.now();
      const params = args[0];
      let response = null;
      let ok = true;
      let error = null;
      try {
        response = await handler.apply(this, args);
        if (response && response.isError) {
          ok = false;
          const text = response.content && response.content[0] && response.content[0].text;
          if (text) error = String(text).slice(0, 500);
        }
        return response;
      } catch (err) {
        ok = false;
        error = err && err.message ? err.message : String(err);
        throw err;
      } finally {
        const event = {
          ts: new Date().toISOString(),
          kind: "tool_call",
          agent_id: writer.agentId,
          tool: toolName,
          params,
          response,
          latency_ms: Date.now() - start,
          ok,
        };
        if (error) event.error = error;
        writer.write(event);
      }
    };
  }
}

// ── Read-side helpers used by the CLI ─────────────────────────────

export function listDumps(config = resolveConfig()) {
  const out = [];
  if (!existsSync(config.dir)) return out;
  for (const agent of readdirSync(config.dir)) {
    const agentDir = join(config.dir, agent);
    let st;
    try {
      st = statSync(agentDir);
    } catch {
      continue;
    }
    if (!st.isDirectory()) continue;
    for (const file of readdirSync(agentDir)) {
      if (!file.endsWith(".jsonl")) continue;
      const path = join(agentDir, file);
      const s = statSync(path);
      out.push({
        agent_id: agent,
        date: file.replace(/\.jsonl$/, ""),
        path,
        bytes: s.size,
        mtime: s.mtime.toISOString(),
      });
    }
  }
  out.sort((a, b) => (a.mtime < b.mtime ? 1 : -1));
  return out;
}

export function todayDumpPath(agentId, config = resolveConfig()) {
  const date = new Date().toISOString().slice(0, 10);
  return join(config.dir, agentId, `${date}.jsonl`);
}

export { resolveConfig };
