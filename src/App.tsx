import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";
import type { GenerationStats } from "./engine/protocol.ts";
import { DEFAULT_SAMPLING, type SamplingParams } from "./engine/sampler.ts";
import type { ChatMessage } from "./engine/tokenizer.ts";
import { MODELS, type ModelEntry, formatBytes } from "./models.ts";
import { type LoadState, useEngine } from "./useEngine.ts";

interface UiMessage {
  role: "user" | "assistant";
  content: string;
  modelId?: string;
  stats?: GenerationStats;
  pending?: boolean;
  prefill?: { done: number; total: number };
  error?: string;
}

interface Settings extends SamplingParams {
  useSystemPrompt: boolean;
}

const SETTINGS_KEY = "tinylm-settings-v1";
const EXAMPLES = [
  "日本の首都はどこですか？",
  "秋の夜をテーマに短い詩を書いてください。",
  "機械学習とは何か、中学生にもわかるように説明してください。",
  "朝ごはんのおすすめを3つ教えてください。",
];

function loadSettings(): Settings {
  const base: Settings = { ...DEFAULT_SAMPLING, useSystemPrompt: false };
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...base, ...JSON.parse(raw) };
  } catch {
    // storage unavailable
  }
  return base;
}

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

export function App() {
  const engine = useEngine();
  const { load } = engine;
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch {
      // ignore
    }
  }, [settings]);

  useEffect(() => {
    const el = listRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const loadedModel = MODELS.find((m) => m.id === load.modelId) ?? null;
  const ready = load.phase === "ready";

  const runGeneration = useCallback(
    (history: UiMessage[]) => {
      if (!loadedModel) return;
      const chat: ChatMessage[] = [];
      if (settings.useSystemPrompt) chat.push({ role: "system", content: "" });
      for (const m of history) if (!m.error) chat.push({ role: m.role, content: m.content });
      const modelId = loadedModel.id;
      setMessages([...history, { role: "assistant", content: "", modelId, pending: true }]);
      setGenerating(true);
      stickToBottom.current = true;
      const update = (fn: (m: UiMessage) => UiMessage) =>
        setMessages((ms) => {
          const copy = ms.slice();
          copy[copy.length - 1] = fn(copy[copy.length - 1]);
          return copy;
        });
      const { useSystemPrompt: _, ...params } = settings;
      engine.generate(chat, params, {
        onPrefill: (done, total) => update((m) => ({ ...m, prefill: { done, total } })),
        onToken: (text) => update((m) => ({ ...m, content: m.content + text, prefill: undefined })),
        onDone: (text, stats) => {
          update((m) => ({ ...m, content: text, stats, pending: false, prefill: undefined }));
          setGenerating(false);
        },
        onError: (message) => {
          update((m) => ({ ...m, pending: false, error: message }));
          setGenerating(false);
        },
      });
    },
    [engine, loadedModel, settings],
  );

  const send = (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || !ready || generating) return;
    setInput("");
    runGeneration([...messages, { role: "user", content }]);
  };

  const regenerate = () => {
    if (!ready || generating) return;
    const lastUser = messages.map((m) => m.role).lastIndexOf("user");
    if (lastUser < 0) return;
    runGeneration(messages.slice(0, lastUser + 1));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Do not send while an IME composition is active (Japanese input).
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      send();
    }
  };

  const switchModel = (entry: ModelEntry) => {
    if (generating) engine.stop();
    if (entry.id === load.modelId && load.phase !== "error") return;
    engine.loadModel(entry);
  };

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const showPicker = load.phase === "idle" || (load.phase === "error" && !messages.length);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            tl
          </span>
          <span className="brand-name">tiny-lm chat</span>
        </div>
        <div className="topbar-center">
          {load.phase !== "idle" && (
            <label className="model-select">
              <span className="sr-only">モデル</span>
              <select
                value={load.modelId ?? ""}
                disabled={load.phase === "loading"}
                onChange={(e) => {
                  const entry = MODELS.find((m) => m.id === e.target.value);
                  if (entry) switchModel(entry);
                }}
              >
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <div className="topbar-actions">
          {messages.length > 0 && (
            <button
              type="button"
              className="ghost"
              disabled={generating}
              onClick={() => {
                setMessages([]);
                inputRef.current?.focus();
              }}
            >
              新しい会話
            </button>
          )}
          <button
            type="button"
            className={`ghost ${showSettings ? "active" : ""}`}
            aria-expanded={showSettings}
            onClick={() => setShowSettings((v) => !v)}
          >
            設定
          </button>
        </div>
      </header>

      <div className="layout">
        <main className="chat">
          {!hasWebGPU ? (
            <NoWebGPU />
          ) : showPicker ? (
            <ModelPicker cached={engine.cached} onPick={switchModel} error={load.error} />
          ) : (
            <>
              <div
                className="messages"
                ref={listRef}
                onScroll={(e) => {
                  const el = e.currentTarget;
                  stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
                }}
              >
                <div className="messages-inner">
                  {load.phase === "loading" && <LoadingPanel load={load} />}
                  {load.phase === "error" && (
                    <div className="banner error">
                      {load.error}{" "}
                      {loadedModel && (
                        <button type="button" className="link" onClick={() => engine.loadModel(loadedModel)}>
                          もう一度読み込む
                        </button>
                      )}
                    </div>
                  )}
                  {ready && messages.length === 0 && (
                    <div className="empty">
                      <h2>{loadedModel?.name}</h2>
                      <p className="muted">
                        本モデルはブラウザの中だけで動作するので、会話が外部に送信されることはありません。
                      </p>
                      <div className="examples">
                        {EXAMPLES.map((ex) => (
                          <button type="button" key={ex} className="example" onClick={() => send(ex)}>
                            {ex}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {messages.map((m, i) => (
                    <Message
                      key={i}
                      m={m}
                      isLast={i === messages.length - 1}
                      canRegenerate={ready && !generating}
                      onRegenerate={regenerate}
                    />
                  ))}
                </div>
              </div>
              <div className="composer-wrap">
                <form
                  className="composer"
                  onSubmit={(e) => {
                    e.preventDefault();
                    send();
                  }}
                >
                  <textarea
                    ref={inputRef}
                    value={input}
                    rows={1}
                    placeholder={ready ? "メッセージを入力（Enter で送信、Shift+Enter で改行）" : "モデルを読み込んでいます…"}
                    disabled={!ready}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                  />
                  {generating ? (
                    <button type="button" className="primary stop" onClick={engine.stop}>
                      停止
                    </button>
                  ) : (
                    <button type="submit" className="primary" disabled={!ready || !input.trim()}>
                      送信
                    </button>
                  )}
                </form>
                <p className="fineprint">
                  小さなモデルなので、事実と異なる内容を出力することがあります。
                  {loadedModel?.notice && (
                    <>
                      {" "}
                      <ModelNotice notice={loadedModel.notice} />
                    </>
                  )}
                </p>
              </div>
            </>
          )}
        </main>

        {showSettings && (
          <SettingsPanel
            settings={settings}
            setSettings={setSettings}
            onClose={() => setShowSettings(false)}
            engine={engine}
          />
        )}
      </div>
    </div>
  );
}

function Message(props: { m: UiMessage; isLast: boolean; canRegenerate: boolean; onRegenerate: () => void }) {
  const { m, isLast, canRegenerate, onRegenerate } = props;
  if (m.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{m.content}</div>
      </div>
    );
  }
  const model = MODELS.find((x) => x.id === m.modelId);
  const s = m.stats;
  return (
    <div className="msg assistant">
      <div className="msg-meta">{model?.name ?? "assistant"}</div>
      <div className={`bubble ${m.pending ? "streaming" : ""}`}>
        {m.content}
        {m.pending && !m.content && (
          <span className="muted">
            {m.prefill && m.prefill.total > 256
              ? `プロンプトを処理中 ${m.prefill.done}/${m.prefill.total}`
              : "考えています…"}
          </span>
        )}
        {m.pending && <span className="cursor" aria-hidden />}
      </div>
      {m.error && <div className="banner error">{m.error}</div>}
      {s && (
        <div className="msg-footer">
          <span>
            {s.newTokens} トークン · {s.decodeMs > 0 ? (s.newTokens / (s.decodeMs / 1000)).toFixed(1) : "-"} tok/s
          </span>
          {s.stopReason === "stopped" && <span>停止しました</span>}
          {s.stopReason === "length" && <span>最大トークン数に達しました</span>}
          {s.stopReason === "context" && <span>コンテキスト長の上限に達しました</span>}
          {s.droppedMessages > 0 && <span>古い {s.droppedMessages} 件のメッセージを省略しました</span>}
          {isLast && canRegenerate && (
            <button type="button" className="link" onClick={onRegenerate}>
              再生成
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function ModelPicker(props: {
  cached: Record<string, { bytes: number; complete: boolean }>;
  onPick: (m: ModelEntry) => void;
  error: string | null;
}) {
  return (
    <div className="picker">
      <div className="picker-intro">
        <h1>ちょっとアホな言語モデルとのチャットです</h1>
      </div>
      {props.error && <div className="banner error">{props.error}</div>}
      <div className="cards">
        {MODELS.map((m) => {
          const isCached = props.cached[m.repo]?.complete ?? false;
          return (
            <div className="card" key={m.id}>
              <div className="card-head">
                <h3>{m.name}</h3>
                {isCached ? <span className="badge ok">キャッシュ済み</span> : null}
              </div>
              <p className="card-summary">{m.summary}</p>
              <ul className="card-details">
                {m.details.map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
              <div className="card-foot">
                <span className="muted">
                  {isCached ? "ダウンロード不要" : `ダウンロード ${formatBytes(m.weightBytes)}`} · GPU メモリ約{" "}
                  {formatBytes(m.weightBytes + 250e6)}
                </span>
                <button type="button" className="primary" onClick={() => props.onPick(m)}>
                  このモデルで始める
                </button>
              </div>
              <a className="card-link" href={`https://huggingface.co/${m.repo}`} target="_blank" rel="noreferrer">
                {m.repo}
              </a>
              {m.notice && <ModelNotice notice={m.notice} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ModelNotice({ notice }: { notice: NonNullable<ModelEntry["notice"]> }) {
  return (
    <a className="notice" href={notice.href} target="_blank" rel="noreferrer">
      {notice.text}
    </a>
  );
}

function LoadingPanel({ load }: { load: LoadState }) {
  const pct = load.totalBytes ? Math.min(100, (load.doneBytes / load.totalBytes) * 100) : 0;
  const elapsed = (performance.now() - load.startedAt) / 1000;
  const rate = elapsed > 1 ? load.doneBytes / elapsed : 0;
  const remaining = rate > 0 ? (load.totalBytes - load.doneBytes) / rate : 0;
  const model = MODELS.find((m) => m.id === load.modelId);
  return (
    <div className="loading">
      <h2>{model?.name} を読み込んでいます</h2>
      <p className="muted">{load.status}</p>
      <div className="progress" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-bar" style={{ width: `${pct}%` }} />
      </div>
      <p className="muted small">
        {formatBytes(load.doneBytes)} / {formatBytes(load.totalBytes)}
        {rate > 0 && !load.fromCache && ` · ${formatBytes(rate)}/s · 残り約 ${formatDuration(remaining)}`}
      </p>
    </div>
  );
}

function formatDuration(sec: number): string {
  if (!Number.isFinite(sec)) return "-";
  if (sec < 60) return `${Math.ceil(sec)} 秒`;
  return `${Math.floor(sec / 60)} 分 ${Math.ceil(sec % 60)} 秒`;
}

function NoWebGPU() {
  return (
    <div className="picker">
      <div className="picker-intro">
        <h1>WebGPU が使えません</h1>
        <p className="muted">
          このサイトは WebGPU でモデルを動かします。最新の Chrome / Edge（デスクトップ）、または WebGPU
          に対応した Safari・Firefox でお試しください。
        </p>
      </div>
    </div>
  );
}

function SettingsPanel(props: {
  settings: Settings;
  setSettings: (fn: (s: Settings) => Settings) => void;
  onClose: () => void;
  engine: ReturnType<typeof useEngine>;
}) {
  const { settings, setSettings, engine } = props;
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => setSettings((s) => ({ ...s, [k]: v }));
  const totalCached = Object.values(engine.cached).reduce((a, b) => a + b.bytes, 0);
  return (
    <aside className="settings" aria-label="設定">
      <div className="settings-head">
        <h2>生成の設定</h2>
        <button type="button" className="ghost" onClick={props.onClose}>
          閉じる
        </button>
      </div>
      <Slider label="Temperature" hint="0 で常に最も確率の高いトークンを選びます" min={0} max={2} step={0.05} value={settings.temperature} onChange={(v) => set("temperature", v)} />
      <Slider label="Top-p" min={0.05} max={1} step={0.05} value={settings.topP} onChange={(v) => set("topP", v)} />
      <Slider label="Top-k" hint="0 で無効" min={0} max={200} step={1} value={settings.topK} onChange={(v) => set("topK", v)} />
      <Slider label="繰り返しペナルティ" hint="1 で無効。生成済みのトークンに適用します" min={1} max={1.5} step={0.01} value={settings.repetitionPenalty} onChange={(v) => set("repetitionPenalty", v)} />
      <Slider label="最大生成トークン数" min={16} max={1024} step={16} value={settings.maxNewTokens} onChange={(v) => set("maxNewTokens", v)} />
      <label className="check">
        <input type="checkbox" checked={settings.useSystemPrompt} onChange={(e) => set("useSystemPrompt", e.target.checked)} />
        <span>
          システムプロンプトを付ける
          <small className="muted">「以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。」</small>
        </span>
      </label>
      <button type="button" className="ghost" onClick={() => setSettings((s) => ({ ...s, ...DEFAULT_SAMPLING, useSystemPrompt: false }))}>
        既定値に戻す
      </button>

      <div className="settings-section">
        <h3>キャッシュ</h3>
        <p className="muted small">
          ダウンロード済みの重み: {formatBytes(totalCached)}
          {engine.storage.quota > 0 && `（ブラウザの保存容量 ${formatBytes(engine.storage.usage)} / ${formatBytes(engine.storage.quota)}）`}
        </p>
        <button type="button" className="ghost danger" disabled={totalCached === 0} onClick={engine.clearCache}>
          キャッシュを削除
        </button>
      </div>
      {engine.load.adapter && (
        <div className="settings-section">
          <h3>GPU</h3>
          <p className="muted small">{engine.load.adapter}</p>
        </div>
      )}
    </aside>
  );
}

function Slider(props: {
  label: string;
  hint?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="slider">
      <span className="slider-row">
        <span>{props.label}</span>
        <output>{props.value}</output>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      {props.hint && <small className="muted">{props.hint}</small>}
    </label>
  );
}
