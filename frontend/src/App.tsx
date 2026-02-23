import { useEffect, useMemo, useState } from "react";
import {
  createRun,
  fetchDefaults,
  fetchRunLog,
  getAuthTokenStatus,
  infer,
  listRuns,
  setAuthToken,
  uploadDataset
} from "./api";

type Defaults = Record<string, any>;

type RunRow = {
  run_id: string;
  status: string;
  detail?: string;
  result?: any;
};

const sectionStyle: React.CSSProperties = {
  background: "#ffffff",
  borderRadius: 16,
  padding: 20,
  boxShadow: "0 14px 40px rgba(15, 23, 42, 0.08)",
  border: "1px solid #e2e8f0"
};

function Input({
  label,
  value,
  onChange,
  placeholder
}: {
  label: string;
  value: string | number;
  onChange: (val: string) => void;
  placeholder?: string;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span style={{ fontSize: 14, color: "#475569" }}>{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          padding: "10px 12px",
          borderRadius: 10,
          border: "1px solid #cbd5e1",
          fontSize: 14,
          background: "#f8fafc"
        }}
      />
    </label>
  );
}

function TextArea({
  label,
  value,
  onChange,
  rows = 4,
  placeholder
}: {
  label: string;
  value: string;
  onChange: (val: string) => void;
  rows?: number;
  placeholder?: string;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span style={{ fontSize: 14, color: "#475569" }}>{label}</span>
      <textarea
        rows={rows}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          padding: "10px 12px",
          borderRadius: 12,
          border: "1px solid #cbd5e1",
          fontSize: 14,
          background: "#f8fafc",
          fontFamily: "inherit",
          resize: "vertical"
        }}
      />
    </label>
  );
}

function App() {
  const [defaults, setDefaults] = useState<Defaults>({});
  const [configOverrides, setConfigOverrides] = useState<Defaults>({});
  const [datasetPath, setDatasetPath] = useState("");
  const [datasetInfo, setDatasetInfo] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runId, setRunId] = useState("");

  const [tokenInput, setTokenInput] = useState("");
  const [tokenStatus, setTokenStatus] = useState<string>("not set");

  const [plainModelOnly, setPlainModelOnly] = useState(false);

  const [inferenceRunId, setInferenceRunId] = useState("");
  const [inferenceInput, setInferenceInput] = useState("");
  const [inferenceOutput, setInferenceOutput] = useState("");
  const [inferenceStatus, setInferenceStatus] = useState<string | null>(null);

  const [selectedRunId, setSelectedRunId] = useState("");
  const [runLog, setRunLog] = useState("");

  useEffect(() => {
    fetchDefaults()
      .then((d) => setDefaults(d || {}))
      .catch(() => setDefaults({}));
    getAuthTokenStatus()
      .then((s) => setTokenStatus(s.token_set ? "token set" : "not set"))
      .catch(() => setTokenStatus("unknown"));
    refreshRuns();
  }, []);

  const systemPrompt = useMemo(() => {
    if (configOverrides && "system_prompt" in configOverrides) return configOverrides.system_prompt as string;
    return "";
  }, [configOverrides, defaults]);

  function updateOverride(key: string, value: any) {
    setConfigOverrides((prev) => ({ ...prev, [key]: value }));
  }

  async function refreshRuns() {
    try {
      const data = await listRuns();
      setRuns(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
    }
  }

  async function loadLog(runId: string) {
    try {
      const res = await fetchRunLog(runId);
      setRunLog(res.log);
    } catch (err) {
      setRunLog("log unavailable");
    }
  }

  async function handleUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (!file) return;
    setPending(true);
    try {
      const res = await uploadDataset(file);
      setDatasetPath(res.path);
      setDatasetInfo(`Uploaded ${res.rows} rows -> ${res.path}`);
    } catch (err: any) {
      setDatasetInfo(err?.message ?? "Upload failed");
    } finally {
      setPending(false);
    }
  }

  async function handleSetToken() {
    if (!tokenInput.trim()) {
      setTokenStatus("enter a token first");
      return;
    }
    setPending(true);
    try {
      await setAuthToken(tokenInput.trim());
      setTokenStatus("token set");
      setTokenInput("");
    } catch (err: any) {
      setTokenStatus(err?.message ?? "failed to set token");
    } finally {
      setPending(false);
    }
  }

  async function handleCreateRun() {
    if (!plainModelOnly && !datasetPath) {
      setDatasetInfo("Please upload or provide a dataset path.");
      return;
    }
    setPending(true);
    try {
      const res = await createRun({
        dataset_path: plainModelOnly ? "" : datasetPath,
        run_id: runId || undefined,
        config_overrides: configOverrides
      });
      setDatasetInfo(`Run ${res.run_id} started (${res.status})`);
      setRunId(res.run_id);
      await refreshRuns();
    } catch (err: any) {
      setDatasetInfo(err?.message ?? "Failed to start run");
    } finally {
      setPending(false);
    }
  }

  async function handleInference() {
    if (!inferenceRunId || !inferenceInput) {
      setInferenceStatus("Pick a finished run and type some input.");
      return;
    }
    setInferenceStatus("Generating...");
    try {
      const res = await infer({
        run_id: inferenceRunId,
        input_text: inferenceInput
      });
      setInferenceOutput(res.output);
      setInferenceStatus("Done");
    } catch (err: any) {
      setInferenceStatus(err?.message ?? "Inference failed");
    }
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 18px 48px", display: "flex", flexDirection: "column", gap: 18 }}>
      <header style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <span style={{ color: "#3b82f6", fontWeight: 700, letterSpacing: 0.3 }}>Workbench</span>
        <h1 style={{ margin: 0, fontSize: 32, color: "#0f172a" }}>LLM Finetuning Control Center</h1>
        <p style={{ margin: 0, color: "#475569", maxWidth: 760 }}>
          Pick a base model, craft prompts, upload datasets, launch LoRA/QLoRA runs, and play with the resulting adapters —
          all from one screen. This is a thin UI over the FastAPI backend you can extend.
        </p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 16 }}>
        <section style={sectionStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0, fontSize: 20 }}>Model & Prompt</h2>
            <span style={{ fontSize: 12, color: "#64748b" }}>defaults pulled from backend</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Input
              label="Model (HF id or local path)"
              value={configOverrides.model_name ?? defaults.model_name ?? ""}
              onChange={(v) => updateOverride("model_name", v)}
              placeholder="Qwen/Qwen2.5-1.5B-Instruct"
            />
            <Input
              label="Prompt style"
              value={configOverrides.prompt_style ?? defaults.prompt_style ?? "auto"}
              onChange={(v) => updateOverride("prompt_style", v)}
              placeholder="auto | mistral | llama | plain"
            />
            <Input
              label="Max source length"
              value={configOverrides.max_source_length ?? defaults.max_source_length ?? 2048}
              onChange={(v) => updateOverride("max_source_length", Number(v))}
            />
            <Input
              label="Max target length"
              value={configOverrides.max_target_length ?? defaults.max_target_length ?? 512}
              onChange={(v) => updateOverride("max_target_length", Number(v))}
            />
          </div>
          <div style={{ marginTop: 12 }}>
            <TextArea
              label="System prompt"
              value={systemPrompt}
              onChange={(v) => updateOverride("system_prompt", v)}
              rows={6}
              placeholder={"Describe the task and output format, e.g.:\n- You are an extractor...\n- Return JSON with fields..."}
            />
          </div>
        </section>

        <section style={sectionStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h2 style={{ margin: 0, fontSize: 20 }}>Dataset</h2>
            <small style={{ color: "#475569" }}>input_text / output_text columns</small>
          </div>
          {plainModelOnly ? (
            <p style={{ margin: 0, color: "#475569" }}>Plain model mode: no dataset needed.</p>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <input type="file" accept=".csv" onChange={handleUpload} disabled={pending} />
              <Input label="or use existing path" value={datasetPath} onChange={setDatasetPath} placeholder="data/data.csv" />
              <p style={{ margin: 0, color: "#475569", minHeight: 24 }}>{datasetInfo}</p>
            </div>
          )}
        </section>
      </div>

      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0, fontSize: 20 }}>Access</h2>
          <span style={{ fontSize: 12, color: "#64748b" }}>Hugging Face token for gated models</span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Input label="HF token" value={tokenInput} onChange={setTokenInput} placeholder="hf_xxx" />
          <button
            onClick={handleSetToken}
            disabled={pending}
            style={{
              minWidth: 140,
              border: "none",
              borderRadius: 12,
              background: "linear-gradient(135deg, #6d28d9, #9333ea)",
              color: "white",
              fontWeight: 700,
              cursor: "pointer",
              height: 44,
              marginTop: 22
            }}
          >
            Save token
          </button>
          <span style={{ color: "#475569" }}>Status: {tokenStatus}</span>
        </div>
      </section>

      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0, fontSize: 20 }}>Training parameters</h2>
          <span style={{ fontSize: 12, color: "#64748b" }}>LoRA + QLoRA ready</span>
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
          <input
            type="checkbox"
            checked={plainModelOnly}
            onChange={(e) => setPlainModelOnly(e.target.checked)}
          />
          <span style={{ color: "#0f172a", fontWeight: 600 }}>Plain model only (no finetune)</span>
          <span style={{ color: "#475569" }}>Use base model for inference; skips training.</span>
        </label>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
          <Input
            label="Epochs"
            value={configOverrides.num_train_epochs ?? defaults.num_train_epochs ?? 3}
            onChange={(v) => updateOverride("num_train_epochs", Number(v))}
          />
          <Input
            label="LR"
            value={configOverrides.learning_rate ?? defaults.learning_rate ?? 2e-4}
            onChange={(v) => updateOverride("learning_rate", Number(v))}
          />
          <Input
            label="Batch size / device"
            value={configOverrides.per_device_train_batch_size ?? defaults.per_device_train_batch_size ?? 1}
            onChange={(v) => updateOverride("per_device_train_batch_size", Number(v))}
          />
          <Input
            label="LoRA r"
            value={configOverrides.lora_r ?? defaults.lora_r ?? 4}
            onChange={(v) => updateOverride("lora_r", Number(v))}
          />
          <Input
            label="LoRA alpha"
            value={configOverrides.lora_alpha ?? defaults.lora_alpha ?? 8}
            onChange={(v) => updateOverride("lora_alpha", Number(v))}
          />
          <Input
            label="LoRA dropout"
            value={configOverrides.lora_dropout ?? defaults.lora_dropout ?? 0.05}
            onChange={(v) => updateOverride("lora_dropout", Number(v))}
          />
          <Input
            label="4-bit?"
            value={configOverrides.load_in_4bit ?? defaults.load_in_4bit ?? true}
            onChange={(v) => updateOverride("load_in_4bit", v === "true" || v === "1")}
          />
        </div>
        <div style={{ display: "flex", gap: 12, marginTop: 14 }}>
          <Input label="Optional run id" value={runId} onChange={setRunId} placeholder="auto-generated if blank" />
          <button
            onClick={handleCreateRun}
            disabled={pending}
            style={{
              minWidth: 180,
              border: "none",
              borderRadius: 12,
              background: "linear-gradient(135deg, #2563eb, #0ea5e9)",
              color: "white",
              fontWeight: 700,
              cursor: "pointer"
            }}
          >
            {pending ? "Starting..." : "Launch run"}
          </button>
        </div>
      </section>

      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0, fontSize: 20 }}>Runs</h2>
          <button
            onClick={refreshRuns}
            style={{
              border: "1px solid #cbd5e1",
              background: "#f8fafc",
              borderRadius: 10,
              padding: "6px 12px",
              cursor: "pointer"
            }}
          >
            Refresh
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
          {runs.map((r) => (
            <div key={r.run_id} style={{ padding: 12, borderRadius: 12, border: "1px solid #e2e8f0", background: "#f8fafc" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <strong>{r.run_id}</strong>
                <span style={{ color: "#0ea5e9", fontWeight: 700 }}>{r.status}</span>
              </div>
              {r.detail && <p style={{ margin: "6px 0", color: "#475569" }}>{r.detail}</p>}
              {r.result?.plain_model ? (
                <p style={{ margin: 0, color: "#475569", fontSize: 13 }}>
                  plain model: {r.result.model_name ?? r.result.output_dir}
                </p>
              ) : (
                r.result?.output_dir && (
                  <p style={{ margin: 0, color: "#475569", fontSize: 13 }}>adapter: {r.result.output_dir}/adapter</p>
                )
              )}
              {r.progress && (
                <p style={{ margin: "4px 0", color: "#16a34a", fontSize: 13 }}>
                  step {r.progress.step ?? "-"} | {r.progress.logs?.loss ? `loss ${r.progress.logs.loss}` : ""}
                </p>
              )}
              <button
                style={{
                  marginTop: 8,
                  border: "none",
                  background: "#0f172a",
                  color: "white",
                  padding: "6px 10px",
                  borderRadius: 10,
                  cursor: "pointer"
                }}
                onClick={() => setInferenceRunId(r.run_id)}
              >
                Use for inference
              </button>
              <button
                style={{
                  marginTop: 8,
                  border: "1px solid #cbd5e1",
                  background: "#fff",
                  color: "#0f172a",
                  padding: "6px 10px",
                  borderRadius: 10,
                  cursor: "pointer",
                  marginLeft: 6
                }}
                onClick={() => {
                  setSelectedRunId(r.run_id);
                  loadLog(r.run_id);
                }}
              >
                View log
              </button>
            </div>
          ))}
        </div>
        {selectedRunId && (
          <div style={{ marginTop: 12 }}>
            <h3 style={{ margin: "8px 0" }}>Log: {selectedRunId}</h3>
            <pre
              style={{
                background: "#0f172a",
                color: "#e2e8f0",
                padding: 12,
                borderRadius: 12,
                maxHeight: 240,
                overflow: "auto",
                whiteSpace: "pre-wrap"
              }}
            >
              {runLog || "No log yet."}
            </pre>
          </div>
        )}
      </section>

      <section style={sectionStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <h2 style={{ margin: 0, fontSize: 20 }}>Inference playground</h2>
          <span style={{ color: "#475569" }}>
            Current run: <strong>{inferenceRunId || "none selected"}</strong>
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <TextArea label="Input text" value={inferenceInput} onChange={setInferenceInput} rows={8} />
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <TextArea label="Model output" value={inferenceOutput} onChange={() => {}} rows={8} />
            <button
              onClick={handleInference}
              style={{
                border: "none",
                borderRadius: 12,
                background: "linear-gradient(135deg, #22c55e, #16a34a)",
                color: "white",
                padding: "10px 14px",
                cursor: "pointer",
                fontWeight: 700
              }}
            >
              Generate
            </button>
            <span style={{ color: "#475569" }}>{inferenceStatus}</span>
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
