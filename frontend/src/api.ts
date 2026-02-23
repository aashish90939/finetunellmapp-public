import axios from "axios";

export type TrainConfigDefaults = Record<string, unknown>;

export async function fetchDefaults() {
  const res = await axios.get<{ defaults: TrainConfigDefaults }>("/api/config/defaults");
  return res.data.defaults;
}

export async function setAuthToken(token: string) {
  const res = await axios.post("/api/auth/token", { token });
  return res.data as { ok: boolean };
}

export async function getAuthTokenStatus() {
  const res = await axios.get("/api/auth/token");
  return res.data as { token_set: boolean };
}

export async function uploadDataset(file: File) {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post("/api/datasets/upload", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data as { dataset_id: string; path: string; rows: number };
}

export async function createRun(payload: {
  dataset_path: string;
  run_id?: string;
  config_overrides: Record<string, unknown>;
}) {
  const res = await axios.post("/api/runs", payload);
  return res.data as { run_id: string; status: string; detail?: string };
}

export async function listRuns() {
  const res = await axios.get("/api/runs");
  return res.data as Array<{ run_id: string; status: string; detail?: string; result?: any; progress?: any }>;
}

export async function infer(payload: {
  run_id: string;
  input_text: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
}) {
  const res = await axios.post("/api/infer", payload);
  return res.data as { output: string };
}

export async function fetchRunLog(runId: string, maxBytes = 4000) {
  const res = await axios.get(`/api/runs/${runId}/log`, { params: { max_bytes: maxBytes } });
  return res.data as { log: string };
}
