"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE =
  "Compare GLP-1 receptor agonists and SGLT2 inhibitors for adults with type 2 diabetes and established cardiovascular disease. What do the major outcomes trials show, which populations were studied, and which class is more suitable as the first add-on in a research briefing for a clinical-operations team?";

type Doc = {
  id: string;
  title: string;
  source: string;
  domain?: string | null;
};

type Step = { type: string; message: string; data?: Record<string, unknown> };

type Source = {
  title: string;
  url?: string | null;
  chunk_id?: string | null;
  source?: string;
  note?: string;
};

function renderMarkdown(md: string): string {
  const escaped = md
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  const lines = escaped.split("\n");
  const out: string[] = [];
  let inTable = false;
  let inList = false;
  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };
  const closeTable = () => {
    if (inTable) {
      out.push("</tbody></table>");
      inTable = false;
    }
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("|")) {
      closeList();
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^:?-+:?$/.test(c))) continue;
      if (!inTable) {
        out.push("<table><tbody>");
        inTable = true;
        out.push("<tr>" + cells.map((c) => `<th>${c}</th>`).join("") + "</tr>");
      } else {
        out.push("<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>");
      }
      continue;
    }
    closeTable();
    if (/^### /.test(line)) {
      closeList();
      out.push(`<h3>${line.slice(4)}</h3>`);
    } else if (/^## /.test(line)) {
      closeList();
      out.push(`<h2>${line.slice(3)}</h2>`);
    } else if (/^# /.test(line)) {
      closeList();
      out.push(`<h1>${line.slice(2)}</h1>`);
    } else if (/^[-*] /.test(line)) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${line.slice(2)}</li>`);
    } else if (!line.trim()) {
      closeList();
      out.push("");
    } else {
      closeList();
      const withCitations = line.replace(/\[(\d+)\]/g, "<sup>[$1]</sup>");
      out.push(`<p>${withCitations}</p>`);
    }
  }
  closeList();
  closeTable();
  return out.join("\n");
}

export default function HomePage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [markdown, setMarkdown] = useState("");
  const [steps, setSteps] = useState<Step[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [chunks, setChunks] = useState<Source[]>([]);
  const [critic, setCritic] = useState<string>("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [ingestMsg, setIngestMsg] = useState("");
  const reportRef = useRef<HTMLDivElement>(null);

  async function refreshDocs() {
    try {
      const res = await fetch(`${API}/documents`);
      if (res.ok) setDocs(await res.json());
    } catch {
      /* API may still be booting */
    }
  }

  useEffect(() => {
    refreshDocs();
  }, []);

  useEffect(() => {
    reportRef.current?.scrollTo({ top: reportRef.current.scrollHeight, behavior: "smooth" });
  }, [markdown, steps]);

  const html = useMemo(() => renderMarkdown(markdown), [markdown]);

  async function runQuery(prompt: string) {
    if (!prompt.trim() || running) return;
    setRunning(true);
    setSteps([]);
    setMarkdown("");
    setSources([]);
    setChunks([]);
    setCritic("");
    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: prompt, conversation_id: conversationId }),
      });
      if (!res.body) throw new Error("No stream");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6)) as Step;
          setSteps((prev) => [...prev, event]);
          if (event.type === "critic") {
            const passed = Boolean(event.data && (event.data as { pass_check?: boolean }).pass_check);
            setCritic(passed ? "pass" : "fail");
          }
          if (event.type === "final") {
            const data = (event.data || {}) as {
              conversation_id?: string;
              markdown?: string;
              report?: { citations?: Source[] };
              chunks?: Array<Record<string, unknown>>;
            };
            if (data.conversation_id) setConversationId(data.conversation_id);
            if (data.markdown) setMarkdown(data.markdown);
            setSources(data.report?.citations || []);
            setChunks(
              (data.chunks || []).map((c) => ({
                title: String(c.title || ""),
                url: (c.url as string) || null,
                chunk_id: String(c.chunk_id || ""),
                source: String(c.source || ""),
              })),
            );
          }
        }
      }
    } catch (err) {
      setSteps((prev) => [...prev, { type: "error", message: String(err) }]);
    } finally {
      setRunning(false);
      refreshDocs();
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    runQuery(query);
  }

  async function onUpload(file: File | undefined) {
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    setIngestMsg("Uploading…");
    const res = await fetch(`${API}/ingest/file`, { method: "POST", body });
    const data = await res.json();
    setIngestMsg(data.job_id ? `Queued ${data.job_id.slice(0, 8)}` : "Upload failed");
    setTimeout(refreshDocs, 2500);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Research &amp; Decision Intelligence
          <span>Cited analyst briefings</span>
        </div>
        <small>{conversationId ? `Thread ${conversationId.slice(0, 8)}` : "New thread"}</small>
      </header>

      <aside className="sidebar">
        <h2>Demo corpus</h2>
        <p className="muted">
          Public cardiometabolic evidence is seeded so recruiters can run a real briefing. Upload any other domain —
          the agents stay general.
        </p>
        <button className="example" type="button" onClick={() => setQuery(EXAMPLE)}>
          Load GLP-1 vs SGLT2 example
        </button>
        <label className="drop">
          Add PDF, DOCX, or Markdown
          <input
            type="file"
            hidden
            onChange={(e) => onUpload(e.target.files?.[0])}
            accept=".pdf,.docx,.md,.txt,.html"
          />
        </label>
        <div className="muted">{ingestMsg}</div>
        <h2>Knowledge base</h2>
        <ul className="doc-list">
          {docs.length === 0 && <li className="muted">Waiting for API / seed ingest…</li>}
          {docs.map((d) => (
            <li key={d.id}>
              <strong>{d.title}</strong>
              <span className="muted">{d.source}</span>
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        <div className="report" ref={reportRef}>
          {markdown ? (
            <div className="report-body" dangerouslySetInnerHTML={{ __html: html }} />
          ) : (
            <div className="empty">
              <h1>Ask a research question</h1>
              <p>
                The router sends specialized agents through hybrid retrieval, live papers, optional trial-catalog SQL,
                a critic, and a cited report. Follow-ups use conversation summary rather than the raw transcript.
              </p>
              <p className="muted">
                Demo question: compare GLP-1 receptor agonists and SGLT2 inhibitors using the seeded outcomes-trial
                notes.
              </p>
            </div>
          )}
        </div>
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={query}
            placeholder="Ask for a cited comparison, a trial listing, or a follow-up such as “what about kidney outcomes?”"
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={running || !query.trim()}>
            {running ? "Running" : "Brief"}
          </button>
        </form>
      </main>

      <aside className="trace">
        <h2>Critic</h2>
        {critic ? (
          <div className={`badge ${critic}`}>{critic === "pass" ? "PASS" : "FAIL · retrieving again"}</div>
        ) : (
          <p className="muted">Verdict appears after the draft.</p>
        )}
        <h2>Agent steps</h2>
        <ul className="steps">
          {steps.length === 0 && <li className="muted">Idle</li>}
          {steps.map((s, i) => (
            <li key={`${s.type}-${i}`}>
              <span className="type">{s.type}</span>
              {s.message}
            </li>
          ))}
        </ul>
        <h2>Citations</h2>
        <ul className="source-list">
          {sources.length === 0 && chunks.length === 0 && <li className="muted">None yet</li>}
          {sources.map((s, i) => (
            <li key={`${s.title}-${i}`}>
              <strong>{s.title}</strong>
              <span className="muted">{s.url || s.chunk_id || s.source}</span>
            </li>
          ))}
        </ul>
        <h2>Retrieved chunks</h2>
        <ul className="source-list">
          {chunks.map((c) => (
            <li key={c.chunk_id || c.title}>
              <strong>{c.title}</strong>
              <span className="muted">{c.chunk_id?.slice(0, 8)} · {c.source}</span>
            </li>
          ))}
        </ul>
      </aside>

      <footer className="footer">
        Analyst briefing only — not medical, legal, or financial advice. Demo trials table is a public subset, not a live
        registry. Private knowledge stays in your corpus; this public repo contains no confidential work data.
      </footer>
    </div>
  );
}
