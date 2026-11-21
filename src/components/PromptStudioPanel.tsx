import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Check, Clock3, Plus, RefreshCcw, Save, Sparkles, Trash2 } from "lucide-react";
import {
  axiomApi,
  type PromptTemplate,
  type PromptTemplateVersion,
  type PromptTemplateVariable,
} from "../api/client";

const EMPTY_VARIABLES = "[]";
const EMPTY_VALUES = "{}";

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString();
}

function parseJsonObject(value: string, label: string): Record<string, unknown> {
  const parsed: unknown = JSON.parse(value);
  if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed as Record<string, unknown>;
}

function parseVariables(value: string): PromptTemplateVariable[] {
  const parsed: unknown = JSON.parse(value);
  if (!Array.isArray(parsed) || parsed.length > 32 || !parsed.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
    throw new Error("Variables must be a JSON array of at most 32 variable objects.");
  }
  return parsed as PromptTemplateVariable[];
}

export default function PromptStudioPanel() {
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [draft, setDraft] = useState({ name: "", description: "", body: "", variables: EMPTY_VARIABLES });
  const [valuesJson, setValuesJson] = useState(EMPTY_VALUES);
  const [rendered, setRendered] = useState("");
  const [versions, setVersions] = useState<PromptTemplateVersion[]>([]);
  const [search, setSearch] = useState("");
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const selected = useMemo(
    () => templates.find((template) => template.id === selectedId) ?? null,
    [templates, selectedId],
  );

  const refresh = async (preferredId?: string) => {
    setLoading(true);
    setError("");
    try {
      const result = await axiomApi.promptTemplates({ limit: 100, offset: 0, search });
      const ids = result.templates.map((item) => item.id);
      setTemplates(result.templates.map((item) => ({ ...item, body: "", variables: [] } as PromptTemplate)));
      const nextId = preferredId && ids.includes(preferredId) ? preferredId : ids.includes(selectedId) ? selectedId : "";
      setSelectedId(nextId);
      if (nextId) {
        const detail = await axiomApi.promptTemplate(nextId);
        setDraft({
          name: detail.template.name,
          description: detail.template.description,
          body: detail.template.body,
          variables: JSON.stringify(detail.template.variables, null, 2),
        });
        setTemplates((current) => current.map((item) => item.id === nextId ? detail.template : item));
        const history = await axiomApi.promptTemplateVersions(nextId);
        setVersions(history.versions);
      } else {
        setDraft({ name: "", description: "", body: "", variables: EMPTY_VARIABLES });
        setVersions([]);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template inventory is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const chooseTemplate = async (id: string) => {
    setSelectedId(id);
    setError("");
    setNotice("");
    setRendered("");
    setBusy(true);
    try {
      const [detail, history] = await Promise.all([
        axiomApi.promptTemplate(id),
        axiomApi.promptTemplateVersions(id),
      ]);
      setDraft({
        name: detail.template.name,
        description: detail.template.description,
        body: detail.template.body,
        variables: JSON.stringify(detail.template.variables, null, 2),
      });
      setTemplates((current) => current.map((item) => item.id === id ? detail.template : item));
      setVersions(history.versions);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template could not be loaded.");
    } finally {
      setBusy(false);
    }
  };

  const createTemplate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await axiomApi.createPromptTemplate({
        name,
        description: "New workspace prompt template",
        body: "Write a response about {{topic}}.",
        variables: [{ name: "topic", type: "string", required: true }],
      });
      setNewName("");
      await refresh(result.template.id);
      setNotice(`Created ${result.template.name} at version 1.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template could not be created.");
    } finally {
      setBusy(false);
    }
  };

  const saveTemplate = async () => {
    if (!selected) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const variables = parseVariables(draft.variables);
      const result = await axiomApi.updatePromptTemplate(selected.id, {
        expected_version: selected.version,
        name: draft.name,
        description: draft.description,
        body: draft.body,
        variables,
      });
      setTemplates((current) => current.map((item) => item.id === selected.id ? result.template : item));
      setDraft({
        name: result.template.name,
        description: result.template.description,
        body: result.template.body,
        variables: JSON.stringify(result.template.variables, null, 2),
      });
      const history = await axiomApi.promptTemplateVersions(selected.id);
      setVersions(history.versions);
      setNotice(`Saved ${result.template.name} as version ${result.template.version}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const renderTemplate = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected) return;
    setBusy(true);
    setError("");
    setNotice("");
    setRendered("");
    try {
      const result = await axiomApi.renderPromptTemplate(selected.id, { values: parseJsonObject(valuesJson, "Values") });
      setRendered(result.rendered);
      setNotice(`Rendered revision ${result.version}. No model or tool was invoked.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template could not be rendered.");
    } finally {
      setBusy(false);
    }
  };

  const deleteTemplate = async () => {
    if (!selected || !window.confirm(`Delete prompt template “${selected.name}”? This removes its retained revisions too.`)) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await axiomApi.deletePromptTemplate(selected.id, selected.version);
      setSelectedId("");
      setRendered("");
      await refresh();
      setNotice(`Deleted ${selected.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt template could not be deleted.");
    } finally {
      setBusy(false);
    }
  };

  const loadRevision = async (revision: number) => {
    if (!selected) return;
    setBusy(true);
    setError("");
    try {
      const result = await axiomApi.promptTemplateRevision(selected.id, revision);
      setDraft({
        name: result.revision.name,
        description: result.revision.description,
        body: result.revision.body,
        variables: JSON.stringify(result.revision.variables, null, 2),
      });
      setNotice(`Loaded revision ${revision} for review. Saving creates a new revision; it does not rewrite history.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Prompt revision could not be loaded.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="prompt-studio-layout">
      <section className="surface prompt-studio-library">
        <div className="surface-head">
          <div><p className="surface-eyebrow">WORKSPACE-LOCAL PROMPTS</p><h2>Template library</h2></div>
          <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || busy}><RefreshCcw size={14} /> Refresh</button>
        </div>
        <form className="prompt-studio-create" onSubmit={(event) => void createTemplate(event)}>
          <label className="field" htmlFor="prompt-template-new-name"><span className="field-label-row">Create template</span><input id="prompt-template-new-name" value={newName} onChange={(event) => setNewName(event.target.value)} maxLength={80} placeholder="e.g. Release notes" required /></label>
          <button type="submit" className="button primary-button" disabled={busy || !newName.trim()}><Plus size={14} /> Create</button>
        </form>
        <label className="field" htmlFor="prompt-template-search"><span className="field-label-row">Search templates</span><input id="prompt-template-search" value={search} onChange={(event) => setSearch(event.target.value)} maxLength={120} placeholder="Name, description or body" /></label>
        <div className="prompt-studio-template-list" role="list" aria-label="Prompt templates">
          {templates.map((template) => (
            <button type="button" className={`prompt-studio-template ${template.id === selectedId ? "selected" : ""}`} key={template.id} onClick={() => void chooseTemplate(template.id)} aria-pressed={template.id === selectedId}>
              <span><b>{template.name}</b><small>Version {template.version} · {template.variable_count ?? template.variables?.length ?? 0} typed variables</small></span>
              {template.id === selectedId && <Check size={15} aria-hidden="true" />}
            </button>
          ))}
          {loading && <p className="loading-state" role="status">Loading templates…</p>}
          {!loading && templates.length === 0 && <p className="quiet-note">No templates yet. Create one to start a workspace-local prompt library.</p>}
        </div>
      </section>

      <div className="prompt-studio-workspace">
        <section className="surface">
          <div className="surface-head"><div><p className="surface-eyebrow">VERSIONED TEMPLATE EDITOR</p><h2>{selected?.name ?? "New template"}</h2></div><Sparkles size={17} /></div>
          {!selected && <p className="quiet-note">Choose a template from the library, or create one to begin editing.</p>}
          {selected && (
            <>
              <div className="prompt-studio-fields">
                <label className="field" htmlFor="prompt-template-name"><span className="field-label-row">Name</span><input id="prompt-template-name" value={draft.name} onChange={(event) => setDraft((current) => ({ ...current, name: event.target.value }))} maxLength={80} /></label>
                <label className="field" htmlFor="prompt-template-description"><span className="field-label-row">Description</span><input id="prompt-template-description" value={draft.description} onChange={(event) => setDraft((current) => ({ ...current, description: event.target.value }))} maxLength={240} /></label>
                <label className="field prompt-studio-full" htmlFor="prompt-template-body"><span className="field-label-row">Template body · use {{variable_name}} placeholders</span><textarea id="prompt-template-body" value={draft.body} onChange={(event) => setDraft((current) => ({ ...current, body: event.target.value }))} maxLength={65536} rows={8} /></label>
                <label className="field prompt-studio-full" htmlFor="prompt-template-variables"><span className="field-label-row">Typed variable definitions · JSON</span><textarea id="prompt-template-variables" value={draft.variables} onChange={(event) => setDraft((current) => ({ ...current, variables: event.target.value }))} rows={8} spellCheck={false} /></label>
              </div>
              <div className="prompt-studio-editor-actions">
                <button type="button" className="button primary-button" onClick={() => void saveTemplate()} disabled={busy}><Save size={14} /> Save revision</button>
                <button type="button" className="button danger-button" onClick={() => void deleteTemplate()} disabled={busy}><Trash2 size={14} /> Delete template</button>
                <span className="quiet-note">Current version {selected.version} · SHA-256 {selected.content_sha256.slice(0, 12)}…</span>
              </div>
              <form className="prompt-studio-render" onSubmit={(event) => void renderTemplate(event)}>
                <div className="surface-head"><div><p className="surface-eyebrow">SAFE LOCAL SUBSTITUTION</p><h3>Preview prompt</h3></div><Sparkles size={15} /></div>
                <label className="field" htmlFor="prompt-template-values"><span className="field-label-row">Variable values · JSON scalar values</span><textarea id="prompt-template-values" value={valuesJson} onChange={(event) => setValuesJson(event.target.value)} rows={4} spellCheck={false} /></label>
                <button type="submit" className="button secondary-button" disabled={busy}><Sparkles size={14} /> Render preview</button>
                {rendered && <pre className="prompt-studio-output" aria-label="Rendered prompt">{rendered}</pre>}
                <p className="quiet-note">Rendering only performs bounded literal substitution. It does not execute template code, invoke a model, call a tool, or send data to a provider.</p>
              </form>
              <div className="prompt-studio-history">
                <div className="surface-head"><div><p className="surface-eyebrow">IMMUTABLE REVISION HISTORY</p><h3>Revisions</h3></div><Clock3 size={15} /></div>
                {versions.map((version) => (
                  <div className="prompt-studio-revision" key={version.version}>
                    <span><b>Version {version.version}</b><small>{version.name} · {formatDate(version.updated_at)}</small></span>
                    <code>{version.content_sha256.slice(0, 12)}…</code>
                    <button type="button" className="text-button" onClick={() => void loadRevision(version.version)} disabled={busy}>Load</button>
                    {version.current && <span className="status-pill">Current</span>}
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </div>
      {error && <div className="inline-alert warning" role="alert">{error}</div>}
      {notice && <div className="inline-alert success" role="status">{notice}</div>}
    </div>
  );
}
