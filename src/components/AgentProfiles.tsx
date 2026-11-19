import { useEffect, useState, type FormEvent } from "react";
import { Check, Plus, RefreshCcw, Save, ShieldCheck, Trash2 } from "lucide-react";
import { axiomApi, type AgentProfile, type AgentProfileReport } from "../api/client";

export default function AgentProfiles() {
  const [report, setReport] = useState<AgentProfileReport | null>(null);
  const [selectedId, setSelectedId] = useState("default");
  const [draftCapabilities, setDraftCapabilities] = useState<string[]>([]);
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const selected = report?.profiles.find((profile) => profile.id === selectedId) ?? null;
  const dirty = Boolean(selected) && JSON.stringify([...draftCapabilities].sort()) !== JSON.stringify([...(selected?.task_capabilities ?? [])].sort());

  const refresh = async (preferredId?: string) => {
    setLoading(true);
    setError("");
    try {
      const result = await axiomApi.agentProfiles();
      setReport(result);
      const nextId = preferredId && result.profiles.some((profile) => profile.id === preferredId)
        ? preferredId
        : result.profiles.some((profile) => profile.id === selectedId) ? selectedId : "default";
      setSelectedId(nextId);
      setDraftCapabilities(result.profiles.find((profile) => profile.id === nextId)?.task_capabilities ?? []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Agent profile inventory is unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const chooseProfile = (profile: AgentProfile) => {
    setSelectedId(profile.id);
    setDraftCapabilities(profile.task_capabilities);
    setNotice("");
    setError("");
  };

  const toggleCapability = (capability: string, checked: boolean) => {
    setDraftCapabilities((current) => checked
      ? [...new Set([...current, capability])]
      : current.filter((item) => item !== capability));
  };

  const saveCapabilities = async () => {
    if (!selected) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await axiomApi.updateAgentProfile(selected.id, draftCapabilities);
      setReport((current) => current ? { ...current, profiles: current.profiles.map((item) => item.id === result.profile.id ? result.profile : item) } : current);
      setDraftCapabilities(result.profile.task_capabilities);
      setNotice(`Saved ${result.profile.name} at profile version ${result.profile.version}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Agent profile could not be saved.");
    } finally {
      setBusy(false);
    }
  };

  const createProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const result = await axiomApi.createAgentProfile({ name, task_capabilities: [] });
      setNewName("");
      await refresh(result.profile.id);
      setNotice(`Created default-deny profile ${result.profile.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Agent profile could not be created.");
    } finally {
      setBusy(false);
    }
  };

  const deleteProfile = async (profile: AgentProfile) => {
    if (profile.id === "default" || !window.confirm(`Delete agent profile “${profile.name}”?`)) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await axiomApi.deleteAgentProfile(profile.id);
      await refresh("default");
      setNotice(`Deleted ${profile.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Agent profile could not be deleted.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="agent-profiles-layout">
      <section className="surface">
        <div className="surface-head">
          <div><p className="surface-eyebrow">VERSIONED WORKSPACE METADATA</p><h2>Agent profiles</h2></div>
          <button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading || busy}><RefreshCcw size={14} /> Refresh</button>
        </div>
        <p className="agent-profile-disclaimer">Profiles describe task intent. They do not enable tools, grant MCP permissions, or authorize network/process execution.</p>
        <form className="agent-profile-create" onSubmit={(event) => void createProfile(event)}>
          <label className="field" htmlFor="agent-profile-name"><span className="field-label-row">Create profile</span><input id="agent-profile-name" value={newName} onChange={(event) => setNewName(event.target.value)} maxLength={64} placeholder="e.g. Dataset reviewer" required /></label>
          <button type="submit" className="button primary-button" disabled={busy || !newName.trim()}><Plus size={14} /> Create default-deny</button>
        </form>
        <div className="agent-profile-list" role="list" aria-label="Agent profile list">
          {report?.profiles.map((profile) => (
            <div className={`agent-profile-row ${selectedId === profile.id ? "selected" : ""}`} key={profile.id} role="listitem">
              <button type="button" className="agent-profile-select" aria-pressed={selectedId === profile.id} onClick={() => chooseProfile(profile)}>
                <span className="agent-profile-row-name"><b>{profile.name}</b><small>Version {profile.version} · {profile.task_capabilities.length} task labels</small></span>
                {selectedId === profile.id && <Check size={15} aria-hidden="true" />}
              </button>
              {profile.id !== "default" && <button type="button" className="icon-button agent-profile-delete" aria-label={`Delete ${profile.name}`} onClick={() => void deleteProfile(profile)} disabled={busy}><Trash2 size={14} /></button>}
            </div>
          ))}
          {loading && <p className="loading-state" role="status">Loading profiles…</p>}
        </div>
      </section>

      <section className="surface agent-profile-editor">
        <div className="surface-head"><div><p className="surface-eyebrow">TASK CAPABILITY LABELS</p><h2>{selected?.name ?? "Profile details"}</h2></div><ShieldCheck size={17} /></div>
        {!selected && !loading && <div className="inline-alert warning" role="alert">Select a profile to edit its task labels.</div>}
        {selected && (
          <>
            <div className="agent-profile-facts"><span>PROFILE ID</span><code>{selected.id}</code><span>CURRENT VERSION</span><b>{selected.version}</b></div>
            <fieldset className="agent-profile-capabilities" disabled={busy || loading}>
              <legend>Descriptive tasks available in this profile</legend>
              {(report?.capability_catalog ?? []).map((capability) => (
                <label className="agent-profile-capability" key={capability}>
                  <input type="checkbox" checked={draftCapabilities.includes(capability)} onChange={(event) => toggleCapability(capability, event.target.checked)} />
                  <span><b>{capability}</b><small>{capability.split(".")[0]} task family</small></span>
                </label>
              ))}
            </fieldset>
            <div className="agent-profile-editor-actions">
              <button type="button" className="button primary-button" onClick={() => void saveCapabilities()} disabled={busy || !dirty}><Save size={14} /> {busy ? "Saving…" : "Save profile"}</button>
              <span className="quiet-note">{dirty ? "Unsaved changes" : "Profile version is current"}</span>
            </div>
          </>
        )}
        {error && <div className="inline-alert warning" role="alert">{error}</div>}
        {notice && <div className="inline-alert success" role="status">{notice}</div>}
        {report && <p className="quiet-note">{report.authorization_note}</p>}
      </section>
    </div>
  );
}
