import AxiomOptimizer from "./components/AxiomOptimizer";
import AxiomQuantizer from "./components/AxiomQuantizer";
import ApiPlayground from "./components/ApiPlayground";
import RuntimeConnectorPanel from "./components/RuntimeConnectorPanel";
import AgentProfiles from "./components/AgentProfiles";
import {
  axiomApi,
  type DatasetCleaningPreview,
  type DatasetManifest,
  type DatasetProfile,
  type DatasetQualityResult,
  type HardwareStatus,
  type McpToolPolicy,
  type RuntimeSnapshot,
  type TrainingPlan as TrainingPlanResult,
  type WorkspaceDataset,
  type WorkspaceExperiment,
  type WorkspaceEvaluationRun,
  type WorkspaceEvent,
  type WorkspaceDiagnostics,
  type RoadmapAudit,
  type WorkspaceModel,
  type WorkspaceWorkflow,
  type WorkflowNode,
  type WorkflowNodeType,
  type WorkflowDryRun,
  type WorkflowTemplate,
  type WorkspaceModelCandidate,
  type WorkspaceModelInspection,
} from "./api/client";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  CircleGauge,
  Clock3,
  Command,
  Copy,
  Cpu,
  Database,
  FileCode2,
  GitBranch,
  Gauge,
  Hammer,
  HardDrive,
  KeyRound,
  LayoutDashboard,
  LockKeyhole,
  Menu,
  Moon,
  Network,
  PanelLeftClose,
  Pencil,
  Play,
  Plus,
  RadioTower,
  RefreshCcw,
  Search,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Star,
  Sun,
  Tag,
  TerminalSquare,
  TimerReset,
  Trash2,
  TriangleAlert,
  Upload,
  Wrench,
  X,
  Zap,
} from "lucide-react";

function equalSecrets(a: string, b: string): boolean {
  const left = new TextEncoder().encode(a);
  const right = new TextEncoder().encode(b);

  if (left.length !== right.length) {
    return false;
  }

  let diff = 0;

  for (let i = 0; i < left.length; i += 1) {
    diff |= left[i] ^ right[i];
  }

  return diff === 0;
}


type Page =
  | "dashboard"
  | "models"
  | "datasets"
  | "training"
  | "workflows"
  | "evaluation"
  | "runtime"
  | "mcp"
  | "diagnostics"
  | "api-playground"
  | "agent-profiles"
  | "roadmap"
  | "modules"
  | "logs"
  | "settings"
  | "optimizer"
  | "quantizer";

type Theme = "dark" | "light";
type Tone = "success" | "warning" | "danger" | "neutral" | "info";

type NavItem = {
  id: Page;
  label: string;
  description: string;
  icon: LucideIcon;
  section?: string;
};

type Notice = {
  message: string;
  tone: Tone;
};

const navItems: NavItem[] = [
  {
    id: "optimizer",
    label: "Agent Optimizer",
    description: "Interactive hardware-aware optimization",
    icon: Zap,
    section: "OPTIMIZATION",
  },
  {
    id: "quantizer",
    label: "Quantization Lab",
    description: "Hardware-aware quantization and fit planning",
    icon: SlidersHorizontal,
  },

  {
    id: "dashboard",
    label: "Control center",
    description: "Workspace overview",
    icon: LayoutDashboard,
    section: "WORKSPACE",
  },
  {
    id: "models",
    label: "Models",
    description: "Registry and metadata",
    icon: BrainCircuit,
  },
  {
    id: "datasets",
    label: "Datasets",
    description: "Inspect and validate data",
    icon: Database,
  },
  {
    id: "training",
    label: "Training",
    description: "Hardware-aware plans",
    icon: Hammer,
  },
  {
    id: "workflows",
    label: "Workflows",
    description: "Build and validate training pipelines",
    icon: GitBranch,
  },
  {
    id: "evaluation",
    label: "Evaluation",
    description: "Quality gates and reports",
    icon: CircleGauge,
  },
  {
    id: "runtime",
    label: "Runtime",
    description: "Local serving and requests",
    icon: RadioTower,
    section: "OPERATIONS",
  },
  {
    id: "mcp",
    label: "MCP inspector",
    description: "Tools and resources",
    icon: Network,
  },
  {
    id: "diagnostics",
    label: "Diagnostics",
    description: "Health and configuration",
    icon: Wrench,
  },
  {
    id: "api-playground",
    label: "API Playground",
    description: "Inspect allowlisted GET endpoints",
    icon: Network,
  },
  {
    id: "agent-profiles",
    label: "Agent Profiles",
    description: "Versioned task-intent metadata",
    icon: ShieldCheck,
  },
  {
    id: "roadmap",
    label: "Roadmap",
    description: "Scoped feature status and integrity",
    icon: FileCode2,
  },
  {
    id: "modules",
    label: "Module Explorer",
    description: "Find GUI surfaces, CLI command families, and API reference",
    icon: Network,
  },
  {
    id: "logs",
    label: "Logs",
    description: "Runtime event stream",
    icon: TerminalSquare,
  },
  {
    id: "settings",
    label: "Settings",
    description: "Workspace and security",
    icon: Settings,
    section: "SYSTEM",
  },
];

const pageMeta: Record<
  Page,
  { label: string; kicker: string; description: string }
> = {
  optimizer: {
    label: "Agent Optimizer",
    kicker: "BETA 5 / MODEL OPTIMIZATION",
    description: "Interactive questionnaire, hardware detection and quantization planning.",
  },
  quantizer: {
    label: "Quantization Lab",
    kicker: "BETA 5 / HARDWARE OPTIMIZATION",
    description: "Choose an agent profile and hardware target to select a model quantization plan.",
  },
  dashboard: {
    label: "Control center",
    kicker: "CONTROL CENTER / LOCAL WORKSPACE",
    description: "A quiet place to see what is ready, what needs attention, and what is still disconnected.",
  },
  models: {
    label: "Models",
    kicker: "MODEL REGISTRY",
    description: "Inspect model metadata and keep local model decisions close to the rest of your stack.",
  },
  datasets: {
    label: "Datasets",
    kicker: "DATA ENGINEERING",
    description: "Validate, clean, and understand the data that feeds your experiments.",
  },
  training: {
    label: "Training",
    kicker: "MODEL OPTIMIZATION",
    description: "Shape a reproducible training plan around the hardware you actually have.",
  },
  workflows: {
    label: "Workflows",
    kicker: "VISUAL PIPELINES",
    description: "Compose and validate a local workflow graph; execution remains disabled until a runner is available.",
  },
  evaluation: {
    label: "Evaluation",
    kicker: "QUALITY GATES",
    description: "Score precomputed outputs, compare local baselines, and review saved plan snapshots.",
  },
  runtime: {
    label: "Runtime",
    kicker: "LOCAL RUNTIME",
    description: "See serving readiness and request flow once the local runtime is connected.",
  },
  mcp: {
    label: "MCP inspector",
    kicker: "MODEL CONTEXT PROTOCOL",
    description: "Grant tools through named profiles; the MCP server denies ungranted tools by default.",
  },
  diagnostics: {
    label: "Diagnostics",
    kicker: "OBSERVABILITY",
    description: "Find configuration drift and local blockers before they become confusing failures.",
  },
  "api-playground": {
    label: "API Playground",
    kicker: "API INSPECTION / READ-ONLY",
    description: "Inspect a fixed list of safe GET endpoints with bounded timeouts and response previews.",
  },
  "agent-profiles": {
    label: "Agent Profiles",
    kicker: "VERSIONED TASK METADATA",
    description: "Create local task profiles while keeping tool permissions and execution scopes separate.",
  },
  roadmap: {
    label: "Roadmap",
    kicker: "DELIVERY STATUS / R001–R120",
    description: "Read the source roadmap, inspect unchecked work, and verify the 120-item scope without guessing status.",
  },
  modules: {
    label: "Module Explorer",
    kicker: "PLATFORM DIRECTORY / GUI · CLI · API",
    description: "A searchable map of AXIOM capabilities, where each one lives, and how to open the available interface.",
  },
  logs: {
    label: "Logs",
    kicker: "EVENT STREAM",
    description: "Trace what the workspace is doing, not just whether it appears to be running.",
  },
  settings: {
    label: "Settings",
    kicker: "SYSTEM / PREFERENCES",
    description: "Tune the local interface and protect the administrator session.",
  },
};

function pageFromLocation(): Page {
  if (typeof window === "undefined") return "dashboard";
  const requested = new URLSearchParams(window.location.search).get("page");
  if (requested === "api-playground") return "api-playground";
  return navItems.some((item) => item.id === requested) ? requested as Page : "dashboard";
}

const platformModules: Array<{
  page: Page;
  title: string;
  description: string;
  state: string;
  tone: Tone;
  icon: LucideIcon;
}> = [
  {
    page: "optimizer",
    title: "Agent Optimizer",
    description: "Questionnaire-driven model sizing, quantization and benchmark planning",
    state: "Beta 5",
    tone: "success",
    icon: Zap,
  },
  {
    page: "models",
    title: "Models",
    description: "Metadata, formats, parameters, quantization",
    state: "CLI surface",
    tone: "neutral",
    icon: BrainCircuit,
  },
  {
    page: "datasets",
    title: "Datasets",
    description: "JSONL inspection, validation, cleaning",
    state: "CLI surface",
    tone: "neutral",
    icon: Database,
  },
  {
    page: "training",
    title: "Training",
    description: "Hardware-aware planning and fit estimates",
    state: "Plan only",
    tone: "warning",
    icon: Hammer,
  },
  {
    page: "workflows",
    title: "Workflows",
    description: "Saved and validated training-pipeline designs",
    state: "Design only",
    tone: "neutral",
    icon: GitBranch,
  },
  {
    page: "runtime",
    title: "Runtime",
    description: "Local inference and serving foundation",
    state: "Not connected",
    tone: "warning",
    icon: RadioTower,
  },
];

const activityBars = [38, 50, 46, 64, 58, 72, 68, 80, 74, 88, 82, 92, 86, 96, 90, 100];

type StoredAdminRecord = {
  version: 1;
  username: string;
  salt: string;
  iv: string;
  ciphertext: string;
  updatedAt: number;
};

const ADMIN_KEY = "axiom-admin-v2";
const LEGACY_ADMIN_KEY = "axiom-admin";
const SESSION_KEY = "axiom-session";
const THEME_KEY = "axiom-theme";

const ADMIN_VERIFIER = "AXIOM-ADMIN-VERIFIER-V1";

function encodeBase64(bytes: Uint8Array): string {
  let binary = "";

  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }

  return btoa(binary);
}

function decodeBase64(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

async function derivePasswordKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(password),
    "PBKDF2",
    false,
    ["deriveKey"],
  );

  return crypto.subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: new Uint8Array(salt).buffer as ArrayBuffer,
      iterations: 310000,
      hash: "SHA-256",
    },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

async function encryptVerifier(password: string, salt: Uint8Array, iv: Uint8Array): Promise<string> {
  const key = await derivePasswordKey(password, salt);
  const encrypted = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: new Uint8Array(iv) },
    key,
    new TextEncoder().encode(ADMIN_VERIFIER),
  );

  return encodeBase64(new Uint8Array(encrypted));
}

async function decryptVerifier(password: string, record: StoredAdminRecord): Promise<boolean> {
  try {
    const key = await derivePasswordKey(password, decodeBase64(record.salt));
    const decrypted = await crypto.subtle.decrypt(
      { name: "AES-GCM", iv: new Uint8Array(decodeBase64(record.iv)) },
      key,
      decodeBase64(record.ciphertext) as unknown as BufferSource,
    );

    return equalSecrets(new TextDecoder().decode(decrypted), ADMIN_VERIFIER);
  } catch {
    return false;
  }
}

function readAdmin(): StoredAdminRecord | null {
  const raw = localStorage.getItem(ADMIN_KEY);

  if (!raw) {
    localStorage.removeItem(LEGACY_ADMIN_KEY);
    return null;
  }

  try {
    const value: unknown = JSON.parse(raw);

    if (
      !value ||
      typeof value !== "object" ||
      (value as Partial<StoredAdminRecord>).version !== 1 ||
      typeof (value as Partial<StoredAdminRecord>).username !== "string" ||
      typeof (value as Partial<StoredAdminRecord>).salt !== "string" ||
      typeof (value as Partial<StoredAdminRecord>).iv !== "string" ||
      typeof (value as Partial<StoredAdminRecord>).ciphertext !== "string"
    ) {
      return null;
    }

    return value as StoredAdminRecord;
  } catch {
    return null;
  }
}

function getStoredTheme(): Theme {
  return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
}

function isAuthenticated(): boolean {
  return sessionStorage.getItem(SESSION_KEY) === "1";
}

function getAdministratorName(): string {
  return readAdmin()?.username ?? "Administrator";
}

async function currentAdministrator(): Promise<string> {
  return getAdministratorName();
}


function getGreeting(): string {
  const hour = new Date().getHours();

  if (hour >= 5 && hour < 12) {
    return "Good morning";
  }

  if (hour >= 12 && hour < 18) {
    return "Good afternoon";
  }

  return "Good evening";
}

async function saveAdministrator(
  username: string,
  password: string,
): Promise<void> {
  const normalizedUsername = username.trim();

  if (normalizedUsername.length < 3 || password.length === 0) {
    throw new Error("Invalid administrator credentials.");
  }

  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await encryptVerifier(password, salt, iv);

  const record: StoredAdminRecord = {
    version: 1,
    username: normalizedUsername,
    salt: encodeBase64(salt),
    iv: encodeBase64(iv),
    ciphertext,
    updatedAt: Date.now(),
  };

  localStorage.setItem(
    ADMIN_KEY,
    JSON.stringify(record),
  );
  localStorage.removeItem(LEGACY_ADMIN_KEY);
}

async function verifyAdministrator(
  username: string,
  password: string,
): Promise<boolean> {
  const admin = readAdmin();

  if (!admin || admin.username !== username.trim()) {
    return false;
  }

  return decryptVerifier(password, admin);
}

function App() {

  const [theme, setTheme] = useState<Theme>(getStoredTheme);
  const [booted, setBooted] = useState(
    () => localStorage.getItem("axiom-booted") === "1",
  );
  const [setupDone, setSetupDone] = useState(
    () => localStorage.getItem("axiom-setup") === "1" && Boolean(localStorage.getItem(ADMIN_KEY)),
  );
  const [authenticated, setAuthenticated] = useState(isAuthenticated);
  const [page, setPage] = useState<Page>(pageFromLocation);
  const [username, setUsername] = useState("Administrator");

  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [changePasswordOpen, setChangePasswordOpen] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  useEffect(() => {
    const restorePage = () => setPage(pageFromLocation());
    window.addEventListener("popstate", restorePage);
    return () => window.removeEventListener("popstate", restorePage);
  }, []);

  useEffect(() => {
    if (!notice) {
      return;
    }

    const timeout = window.setTimeout(() => setNotice(null), 4200);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  useEffect(() => {
    if (!authenticated) {
      return;
    }

    let timeout: number | undefined;
    const logoutForInactivity = () => {
      sessionStorage.removeItem(SESSION_KEY);
      setAuthenticated(false);
      setChangePasswordOpen(false);
      setNotice({ message: "Your session expired after 10 minutes of inactivity.", tone: "warning" });
    };
    const resetTimer = () => {
      if (timeout !== undefined) {
        window.clearTimeout(timeout);
      }

      timeout = window.setTimeout(logoutForInactivity, 10 * 60 * 1000);
    };
    const activityEvents = [
      "mousemove",
      "mousedown",
      "keydown",
      "touchstart",
      "scroll",
      "pointerdown",
    ];

    activityEvents.forEach((event) => window.addEventListener(event, resetTimer));
    resetTimer();

    return () => {
      if (timeout !== undefined) {
        window.clearTimeout(timeout);
      }

      activityEvents.forEach((event) => window.removeEventListener(event, resetTimer));
    };
  }, [authenticated]);

  useEffect(() => {
    if (!authenticated) {
      return;
    }

    currentAdministrator()
      .then(setUsername)
      .catch(() => setUsername("Administrator"));
  }, [authenticated]);

  const setAndPersistTheme = (nextTheme: Theme) => {
    localStorage.setItem(THEME_KEY, nextTheme);
    setTheme(nextTheme);
  };

  const notify = (message: string, tone: Tone = "info") => {
    setNotice({ message, tone });
  };

  const navigate = (nextPage: Page) => {
    if (nextPage !== page) {
      const url = new URL(window.location.href);
      url.searchParams.set("page", nextPage);
      window.history.pushState({ page: nextPage }, "", url);
    }
    setPage(nextPage);
    setMobileNavOpen(false);
  };

  if (!booted) {
    return (
      <Welcome
        theme={theme}
        setTheme={setAndPersistTheme}
        onStart={() => {
          localStorage.setItem("axiom-booted", "1");
          setBooted(true);
        }}
      />
    );
  }

  if (!setupDone) {
    return (
      <Setup
        theme={theme}
        setTheme={setAndPersistTheme}
        onComplete={async (username, password) => {
          await saveAdministrator(username, password);
          localStorage.setItem("axiom-setup", "1");
          setUsername(username.trim());
          sessionStorage.removeItem(SESSION_KEY);

          window.setTimeout(() => {
            setSetupDone(true);
            setAuthenticated(false);
          }, 850);
        }}
      />
    );
  }

  if (!authenticated) {
    return (
      <LoginScreen
        theme={theme}
        setTheme={setAndPersistTheme}
        onLogin={() => {
          sessionStorage.setItem(SESSION_KEY, "1");
          setAuthenticated(true);
        }}
      />
    );
  }

  const logout = () => {
    sessionStorage.removeItem(SESSION_KEY);
    setAuthenticated(false);
    setChangePasswordOpen(false);
  };

  return (
    <div className={`app-shell ${theme} ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        page={page}
        setPage={navigate}
        mobileOpen={mobileNavOpen}
        closeMobile={() => setMobileNavOpen(false)}
        collapsed={sidebarCollapsed}
        toggleCollapsed={() => setSidebarCollapsed((value) => !value)}
      />
      {mobileNavOpen && (
        <button
          type="button"
          className="mobile-scrim"
          aria-label="Close navigation"
          onClick={() => setMobileNavOpen(false)}
        />
      )}

      <div className="main-shell">
        <Topbar
          theme={theme}
          setTheme={setAndPersistTheme}
          page={page}
          username={username}
          onOpenNavigation={() => setMobileNavOpen(true)}
          onLogout={logout}
          onChangePassword={() => setChangePasswordOpen(true)}
          navigate={navigate}
        />

        <main id="main-content" className="page-wrap" tabIndex={-1}>
          <WorkspacePage
            page={page}
            username={username}
            theme={theme}
            setTheme={setAndPersistTheme}
            setPage={navigate}
            onNotify={notify}
            onChangePassword={() => setChangePasswordOpen(true)}
          />
        </main>
      </div>

      {changePasswordOpen && (
        <ChangePasswordDialog
          onClose={() => setChangePasswordOpen(false)}
          onSuccess={() => {
            sessionStorage.removeItem(SESSION_KEY);
            setChangePasswordOpen(false);
            setAuthenticated(false);
            setNotice({ message: "Password changed. Sign in again to continue.", tone: "success" });
          }}
        />
      )}

      {notice && <NoticeToast notice={notice} onDismiss={() => setNotice(null)} />}
    </div>
  );
}

function AxiomMark({ small = false }: { small?: boolean }) {
  return <span className={`axiom-mark ${small ? "small" : ""}`}>A</span>;
}

function ThemeToggle({ theme, setTheme }: { theme: Theme; setTheme: (theme: Theme) => void }) {
  const nextTheme = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      className="icon-button"
      aria-label={`Switch to ${nextTheme} theme`}
      title={`Switch to ${nextTheme} theme`}
      onClick={() => setTheme(nextTheme)}
    >
      {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}

function Welcome({
  theme,
  setTheme,
  onStart,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onStart: () => void;
}) {
  return (
    <div className={`auth-layout ${theme}`}>
      <div className="auth-topline">
        <BrandLockup />
        <ThemeToggle theme={theme} setTheme={setTheme} />
      </div>

      <div className="welcome-grid">
        <section className="welcome-copy" aria-labelledby="welcome-title">
          <StepRail current={1} />
          <p className="eyebrow">FIRST BOOT / AXIOM CONTROL CENTER</p>
          <h1 id="welcome-title">
            Build AI.
            <br />
            <span>Own AI.</span>
          </h1>
          <p className="lede">
            A local-first workspace for models, datasets, training plans, evaluation,
            and runtime decisions, kept close to the engineers making them.
          </p>
          <div className="welcome-actions">
            <button type="button" className="button primary-button" onClick={onStart}>
              Begin setup <ArrowRight size={17} />
            </button>
            <span className="quiet-note">
              <ShieldCheck size={15} /> Stored locally on this installation
            </span>
          </div>
        </section>

        <section className="preview-frame" aria-label="AXIOM workspace preview">
          <div className="preview-header">
            <div className="window-controls" aria-hidden="true"><i /><i /><i /></div>
            <span>AXIOM / CONTROL CENTER</span>
            <StatusPill tone="success">LOCAL</StatusPill>
          </div>
          <div className="preview-body">
            <div className="preview-sidebar">
              <AxiomMark small />
              <span className="preview-line active" />
              <span className="preview-line" />
              <span className="preview-line" />
              <span className="preview-line short" />
            </div>
            <div className="preview-content">
              <div className="preview-kicker">CONTROL CENTER / LOCAL WORKSPACE</div>
              <div className="preview-title">Everything important, in view.</div>
              <div className="preview-status-row">
                <div><span>Workspace</span><b>Ready for setup</b></div>
                <div><span>Runtime</span><b>Not connected</b></div>
              </div>
              <div className="preview-bars" aria-hidden="true">
                {activityBars.slice(0, 10).map((height, index) => (
                  <i key={index} style={{ height: `${height}%` }} />
                ))}
              </div>
              <div className="preview-footer"><span className="status-dot" /> Local, private, inspectable</div>
            </div>
          </div>
        </section>
      </div>

      <div className="auth-footer">
        <span>AXIOM / 0.2.0-beta.5</span>
        <span>LOCAL · PRIVATE · ENGINEERED FOR AI</span>
      </div>
    </div>
  );
}

function BrandLockup() {
  return (
    <div className="brand-lockup">
      <AxiomMark />
      <div>
        <b>AXIOM</b>
        <span>AI ENGINEERING PLATFORM</span>
      </div>
    </div>
  );
}

function StepRail({ current }: { current: 1 | 2 | 3 }) {
  return (
    <div className="step-rail" aria-label={`Setup step ${current} of 3`}>
      {[1, 2, 3].map((step) => (
        <span key={step} className={step <= current ? "active" : ""}>
          0{step}
        </span>
      ))}
    </div>
  );
}

function Setup({
  theme,
  setTheme,
  onComplete,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onComplete: (username: string, password: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [reveal, setReveal] = useState(false);
  const [created, setCreated] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const passwordChecks = getPasswordChecks(password);
  const usernameValid = username.trim().length >= 3;
  const passwordsMatch = password.length > 0 && password === confirm;
  const passwordValid = Object.values(passwordChecks).every(Boolean);
  const canContinue = usernameValid && passwordValid && passwordsMatch && !saving;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!canContinue) {
      return;
    }

    setSaving(true);
    setError("");

    try {
      await onComplete(username, password);
      setCreated(true);
    } catch {
      setError("Unable to save the local administrator. Try again.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={`auth-layout centered ${theme}`}>
      <div className="auth-topline">
        <BrandLockup />
        <ThemeToggle theme={theme} setTheme={setTheme} />
      </div>

      <div className="setup-shell">
        {created ? (
          <div className="success-state">
            <div className="success-mark"><Check size={26} /></div>
            <p className="eyebrow">STEP 3 OF 3 / COMPLETE</p>
            <h1>Administrator created.</h1>
      <p className="lede">Your local workspace is ready. Opening secure sign in...</p>
          </div>
        ) : (
          <form className="auth-form panel-surface" onSubmit={submit}>
            <div className="form-topline">
              <StepRail current={2} />
              <StatusPill tone="neutral">LOCAL ONLY</StatusPill>
            </div>
            <p className="eyebrow">STEP 2 OF 3 / ADMINISTRATOR</p>
            <h1>Create your administrator.</h1>
            <p className="form-intro">This account protects access to the AXIOM Control Center on this installation.</p>

            <Field label="Username" htmlFor="setup-username" hint={usernameValid ? "Username is ready" : "Use at least 3 characters"}>
              <input
                id="setup-username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="administrator"
                autoComplete="username"
                spellCheck={false}
                autoFocus
                aria-invalid={username.length > 0 && !usernameValid}
              />
            </Field>

            <Field
              label="Password"
              htmlFor="setup-password"
              hint="Use a password you do not reuse elsewhere."
              action={
                <button
                  type="button"
                  className="field-action"
                  aria-pressed={reveal}
                  onClick={() => setReveal((value) => !value)}
                >
                  {reveal ? "Hide" : "Show"}
                </button>
              }
            >
              <input
                id="setup-password"
                type={reveal ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Create a secure password"
                autoComplete="new-password"
              />
            </Field>

            <div className="requirements" aria-label="Password requirements">
              <Requirement ok={passwordChecks.length} label="8+ characters" />
              <Requirement ok={passwordChecks.upper} label="Uppercase letter" />
              <Requirement ok={passwordChecks.lower} label="Lowercase letter" />
              <Requirement ok={passwordChecks.number} label="Contains a number" />
            </div>

            <Field
              label="Confirm password"
              htmlFor="setup-confirm"
              hint={confirm.length === 0 ? "Passwords must match" : passwordsMatch ? "Passwords match" : "Passwords do not match"}
              hintTone={passwordsMatch ? "success" : "default"}
            >
              <input
                id="setup-confirm"
                type="password"
                value={confirm}
                onChange={(event) => setConfirm(event.target.value)}
                placeholder="Repeat your password"
                autoComplete="new-password"
                aria-invalid={confirm.length > 0 && !passwordsMatch}
              />
            </Field>

            {error && <InlineAlert tone="danger">{error}</InlineAlert>}

            <button type="submit" className="button primary-button full-width" disabled={!canContinue}>
              {saving ? "Creating account..." : "Create administrator"}
              <ArrowRight size={17} />
            </button>
            <p className="form-note"><LockKeyhole size={14} /> Passwords are never stored in clear text. Only an encrypted verifier is persisted.</p>
          </form>
        )}
      </div>

      <div className="auth-footer"><span>AXIOM / FIRST BOOT</span><span>YOUR WORKSPACE STAYS LOCAL</span></div>
    </div>
  );
}

function getPasswordChecks(password: string) {
  return {
    length: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
  };
}

function Field({
  label,
  htmlFor,
  hint,
  hintTone = "default",
  action,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  hintTone?: "default" | "success";
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <div className="field-label-row"><label htmlFor={htmlFor}>{label}</label>{action}</div>
      <div className="input-shell">{children}</div>
      {hint && <small className={hintTone === "success" ? "success-text" : ""}>{hint}</small>}
    </div>
  );
}

function Requirement({ ok, label }: { ok: boolean; label: string }) {
  return <span className={`requirement ${ok ? "ok" : ""}`}><i>{ok ? <Check size={11} /> : null}</i>{label}</span>;
}

function LoginScreen({
  theme,
  setTheme,
  onLogin,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onLogin: () => void;
}) {
  const [username, setUsername] = useState(getAdministratorName() === "Administrator" ? "" : getAdministratorName());
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reveal, setReveal] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const valid = await verifyAdministrator(username, password);

      if (!valid) {
        setError("Incorrect administrator name or password.");
        return;
      }

      onLogin();
    } catch {
      setError("Unable to authenticate with the local AXIOM account.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`auth-layout login-layout ${theme}`}>
      <div className="auth-topline"><BrandLockup /><ThemeToggle theme={theme} setTheme={setTheme} /></div>
      <div className="login-shell">
        <section className="login-context">
          <p className="eyebrow">SECURE LOCAL CONTROL</p>
          <h1>Welcome back.</h1>
          <p className="lede">Continue to the workspace that keeps your models, data, and decisions close to home.</p>
          <div className="login-facts">
            <div><ShieldCheck size={16} /><span>Local administrator session</span></div>
            <div><HardDrive size={16} /><span>No cloud account required</span></div>
            <div><Clock3 size={16} /><span>10-minute inactivity timeout</span></div>
          </div>
        </section>

        <form className="login-form panel-surface" onSubmit={submit}>
          <div className="form-icon"><KeyRound size={18} /></div>
          <p className="eyebrow">ADMINISTRATOR ACCESS</p>
          <h2>Sign in</h2>
          <p className="form-intro">Use the administrator created during first boot.</p>

          <Field label="Username" htmlFor="login-username">
            <input
              id="login-username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              spellCheck={false}
              placeholder="Administrator"
              autoFocus
            />
          </Field>

          <Field
            label="Password"
            htmlFor="login-password"
            action={
              <button type="button" className="field-action" aria-pressed={reveal} onClick={() => setReveal((value) => !value)}>
                {reveal ? "Hide" : "Show"}
              </button>
            }
          >
            <input
              id="login-password"
              type={reveal ? "text" : "password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              placeholder="Password"
            />
          </Field>

          {error && <InlineAlert tone="danger" role="alert">{error}</InlineAlert>}

          <button className="button primary-button full-width" type="submit" disabled={loading || !username.trim() || !password}>
            {loading ? "Authenticating..." : "Sign in"}
            <ArrowRight size={17} />
          </button>
          <p className="form-note"><ShieldCheck size={14} /> The active session stays in this browser; credentials are not stored in clear text.</p>
        </form>
      </div>
      <div className="auth-footer"><span>AXIOM / CONTROL CENTER</span><span>LOCAL · PRIVATE · INSPECTABLE</span></div>
    </div>
  );
}

function Sidebar({
  page,
  setPage,
  mobileOpen,
  closeMobile,
  collapsed,
  toggleCollapsed,
}: {
  page: Page;
  setPage: (page: Page) => void;
  mobileOpen: boolean;
  closeMobile: () => void;
  collapsed: boolean;
  toggleCollapsed: () => void;
}) {
  return (
    <aside className={`sidebar ${mobileOpen ? "is-open" : ""}`} aria-label="Primary navigation">
      <div className="sidebar-head">
        <button
          type="button"
          className="brand-button"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={toggleCollapsed}
        >
          <AxiomMark />
          <span className="brand-button-copy"><b>AXIOM</b><small>AI ENGINEERING PLATFORM</small></span>
        </button>
        <button type="button" className="icon-button sidebar-close" aria-label="Close navigation" onClick={closeMobile}>
          <PanelLeftClose size={17} />
        </button>
      </div>

      <div className="workspace-switcher">
        <span className="workspace-avatar">A</span>
        <span><b>Local workspace</b><small>Development</small></span>
        <ChevronDown size={15} aria-hidden="true" />
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.id}>
              {item.section && <div className="nav-section">{item.section}</div>}
              <button
                type="button"
                className={`nav-item ${page === item.id ? "active" : ""}`}
                aria-current={page === item.id ? "page" : undefined}
                onClick={() => setPage(item.id)}
              >
                <Icon size={17} strokeWidth={1.8} />
                <span>{item.label}</span>
                {item.id === "diagnostics" && <StatusPill tone="warning">2</StatusPill>}
              </button>
            </div>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <div className="core-status"><span><i className="status-dot" /> UI shell</span><b>READY</b></div>
        <p>AXIOM / BUILD AI. OWN AI.</p>
      </div>
    </aside>
  );
}

function Topbar({
  theme,
  setTheme,
  page,
  username,
  onOpenNavigation,
  onLogout,
  onChangePassword,
  navigate,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  page: Page;
  username: string;
  onOpenNavigation: () => void;
  onLogout: () => void;
  onChangePassword: () => void;
  navigate: (page: Page) => void;
}) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeResult, setActiveResult] = useState(0);
  const [profileOpen, setProfileOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const meta = pageMeta[page];
  const results = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return navItems.slice(0, 5);
    return navItems.filter((item) => `${item.label} ${item.description}`.toLowerCase().includes(normalized));
  }, [query]);

  useEffect(() => setActiveResult(0), [query]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
        setSearchOpen(true);
      }

      if (event.key === "Escape") {
        setSearchOpen(false);
        setProfileOpen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <header className="topbar">
      <div className="topbar-context">
        <button type="button" className="mobile-menu-button icon-button" aria-label="Open navigation" onClick={onOpenNavigation}>
          <Menu size={18} />
        </button>
        <span className="breadcrumb-root">AXIOM</span><ChevronRight size={14} /><span>{meta.label}</span>
      </div>

      <div className="topbar-actions">
        <div className="command-search">
          <Search size={16} />
          <input
            ref={searchRef}
            value={query}
            placeholder="Jump to a workspace view"
            aria-label="Search workspace views"
            aria-expanded={searchOpen}
            aria-controls="workspace-search-results"
            onFocus={() => setSearchOpen(true)}
            onChange={(event) => { setQuery(event.target.value); setSearchOpen(true); }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                setSearchOpen(false);
                (event.target as HTMLInputElement).blur();
              }
              if (event.key === "ArrowDown" && results.length) {
                event.preventDefault();
                setActiveResult((current) => (current + 1) % results.length);
              }
              if (event.key === "ArrowUp" && results.length) {
                event.preventDefault();
                setActiveResult((current) => (current - 1 + results.length) % results.length);
              }
              if (event.key === "Enter" && results[activeResult]) {
                navigate(results[activeResult].id);
                setSearchOpen(false);
                setQuery("");
              }
            }}
          />
          <kbd><Command size={11} /> K</kbd>
          {searchOpen && (
            <div id="workspace-search-results" className="command-menu" role="listbox" aria-label="Workspace views">
              <div className="command-menu-label">NAVIGATE</div>
              {results.length > 0 ? results.map((item, index) => {
                const Icon = item.icon;
                return (
                  <button
                    type="button"
                    role="option"
                    aria-selected={index === activeResult}
                    key={item.id}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => { navigate(item.id); setSearchOpen(false); setQuery(""); }}
                  >
                    <Icon size={16} />
                    <span><b>{item.label}</b><small>{item.description}</small></span>
                    {item.id === page && <Check size={15} />}
                  </button>
                );
              }) : <div className="command-empty">No workspace views match "{query}".</div>}
              <div className="command-hint"><kbd>↑↓</kbd> move <kbd>↵</kbd> open <kbd>esc</kbd> close</div>
            </div>
          )}
        </div>
        <ThemeToggle theme={theme} setTheme={setTheme} />
        <div className="profile-menu">
          <button type="button" className="profile-trigger" aria-expanded={profileOpen} onClick={() => setProfileOpen((value) => !value)}>
            <span className="profile-avatar">{username.charAt(0).toUpperCase()}</span>
            <span className="profile-name">{username}</span>
            <ChevronDown size={14} />
          </button>
          {profileOpen && (
            <div className="profile-dropdown" role="menu">
              <div className="profile-dropdown-head"><small>SIGNED IN AS</small><b>{username}</b></div>
              <button type="button" role="menuitem" onClick={() => { setProfileOpen(false); onChangePassword(); }}><KeyRound size={15} /> Change password</button>
              <button type="button" role="menuitem" className="danger-action" onClick={() => { setProfileOpen(false); onLogout(); }}><ArrowUpRight size={15} /> Sign out</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

function WorkspacePage({
  page,
  username,
  theme,
  setTheme,
  setPage,
  onNotify,
  onChangePassword,
}: {
  page: Page;
  username: string;
  theme: Theme;
  setTheme: (theme: Theme) => void;
  setPage: (page: Page) => void;
  onNotify: (message: string, tone?: Tone) => void;
  onChangePassword: () => void;
}) {
  switch (page) {
    case "optimizer":
      return <AxiomOptimizer />;
    case "quantizer":
      return <AxiomQuantizer />;
    case "dashboard":
      return <Dashboard username={username} setPage={setPage} />;
    case "models":
      return <Models onNotify={onNotify} />;
    case "datasets":
      return <Datasets onNotify={onNotify} />;
    case "training":
      return <Training onNotify={onNotify} />;
    case "workflows":
      return <Workflows onNotify={onNotify} />;
    case "evaluation":
      return <Evaluation onNotify={onNotify} />;
    case "runtime":
      return <Runtime />;
    case "mcp":
      return <MCP onNotify={onNotify} />;
    case "modules":
      return <ModulesPage setPage={setPage} />;
    case "roadmap":
      return <RoadmapPage />;
    case "diagnostics":
      return <Diagnostics onNotify={onNotify} />;
    case "api-playground":
      return <FeaturePage page="api-playground"><ApiPlayground /></FeaturePage>;
    case "agent-profiles":
      return <FeaturePage page="agent-profiles"><AgentProfiles /></FeaturePage>;
    case "logs":
      return <LogsPage />;
    case "settings":
      return <SettingsPage theme={theme} setTheme={setTheme} onChangePassword={onChangePassword} />;
  }
}

function PageHeader({
  page,
  title,
  description,
  actions,
}: {
  page: Page;
  title: ReactNode;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        <p className="eyebrow">{pageMeta[page].kicker}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}

function Dashboard({
  username,
  setPage,
}: {
  username: string;
  setPage: (page: Page) => void;
}) {
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [models, setModels] = useState<WorkspaceModel[] | null>(null);
  const [datasets, setDatasets] = useState<WorkspaceDataset[] | null>(null);
  const [runtime, setRuntime] = useState<RuntimeSnapshot | null>(null);
  const [events, setEvents] = useState<WorkspaceEvent[] | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const refreshWorkspace = async () => {
    setRefreshing(true);
    const [modelResult, datasetResult, runtimeResult, eventResult] = await Promise.allSettled([
      axiomApi.models(),
      axiomApi.datasets(),
      axiomApi.runtime(),
      axiomApi.events(),
    ]);
    if (modelResult.status === "fulfilled") setModels(modelResult.value);
    if (datasetResult.status === "fulfilled") setDatasets(datasetResult.value);
    if (runtimeResult.status === "fulfilled") setRuntime(runtimeResult.value);
    if (eventResult.status === "fulfilled") setEvents(eventResult.value);
    setRefreshing(false);
  };

  useEffect(() => {
    let active = true;
    const checkHealth = () => {
      axiomApi.health()
        .then((data) => {
          if (active) setApiStatus(data.status === "ok" ? "online" : "offline");
        })
        .catch(() => {
          if (active) setApiStatus("offline");
        });
    };
    checkHealth();
    const timer = window.setInterval(checkHealth, 10_000);
    void refreshWorkspace();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const apiLabel = apiStatus === "online" ? "Backend connected" : apiStatus === "checking" ? "Checking backend..." : "Backend unavailable";
  const apiTone: Tone = apiStatus === "online" ? "success" : apiStatus === "checking" ? "neutral" : "warning";
  const runtimeLabel = runtime?.status === "online" ? "Connected" : runtime ? "Not connected" : "Checking";

  return (
    <>
      <PageHeader
        page="dashboard"
        title={<>{getGreeting()}, {username}.</>}
        description="A live view of the local model registry, datasets, hardware, and recent workspace activity."
        actions={<><button type="button" className="button secondary-button" onClick={() => void refreshWorkspace()} disabled={refreshing}><RefreshCcw size={16} /> {refreshing ? "Refreshing" : "Refresh workspace"}</button><button type="button" className="button secondary-button" onClick={() => setPage("diagnostics")}><CircleGauge size={16} /> Diagnostics</button></>}
      />

      <section className="status-strip" aria-label="Workspace status">
        <div className="status-strip-item primary"><StatusPill tone="success">READY</StatusPill><div><b>Local workspace</b><span>Control Center loaded</span></div></div>
        <div className="status-strip-item"><StatusPill tone={apiTone}>{apiStatus === "online" ? "ONLINE" : apiStatus === "checking" ? "CHECKING" : "OFFLINE"}</StatusPill><div><b>AXIOM backend</b><span>{apiLabel}</span></div></div>
        <div className="status-strip-item"><StatusPill tone="neutral">LOCAL</StatusPill><div><b>Session security</b><span>Administrator session active</span></div></div>
      </section>

      <div className="summary-grid">
        <SummaryCard icon={BrainCircuit} label="Model registry" value={models === null ? "—" : String(models.length)} detail={models === null ? "Loading inventory" : models.length ? "Registered locally" : "No models registered"} tone={models?.length ? "success" : "neutral"} />
        <SummaryCard icon={Database} label="Datasets" value={datasets === null ? "—" : String(datasets.length)} detail={datasets === null ? "Loading inventory" : "JSONL files in workspace"} tone={datasets?.length ? "success" : "neutral"} />
        <SummaryCard icon={Cpu} label="Hardware" value={runtime?.hardware.gpu_name || "CPU"} detail={runtime ? `${runtime.hardware.ram_gb.toFixed(1)} GB RAM · ${runtime.hardware.cpu_cores} cores` : "Checking local system"} tone={runtime ? "success" : "neutral"} />
        <SummaryCard icon={RadioTower} label="Inference runtime" value={runtimeLabel} detail="Planning is available; serving is not connected" tone={runtime?.status === "online" ? "success" : "warning"} />
      </div>

      <div className="dashboard-grid">
        <Surface className="activity-surface" title="Recent workspace activity" eyebrow="LOCAL EVENT FEED" action={<StatusPill tone={events?.length ? "success" : "neutral"}>{events ? `${events.length} EVENTS` : "LOADING"}</StatusPill>}>
          {events?.length ? <div className="event-list">{events.slice(0, 5).map((event, index) => <div className="event-row" key={`${event.timestamp}-${index}`}><span className="event-marker" /><div><b>{event.action.replaceAll(".", " / ")}</b><small>{event.detail || "Completed"} · {formatTimestamp(event.timestamp)}</small></div></div>)}</div> : <EmptyState icon={Activity} title={events ? "No workspace activity yet" : "Loading workspace activity"} description={events ? "Register a model, profile a dataset, or create a training plan to start the local event feed." : "The local API event feed is being loaded."} />}
        </Surface>

        <Surface title="Next useful step" eyebrow="ORIENTATION">
          <div className="next-step">
            <div className="next-step-number">01</div>
            <div><h3>{models?.length ? "Build a hardware-aware plan" : "Register a local model"}</h3><p>{models?.length ? "Use the local hardware snapshot to choose a training approach." : "Add a model to the registry, then plan around the hardware available here."}</p><button type="button" className="button small-button primary-button" onClick={() => setPage(models?.length ? "training" : "models")}><ArrowRight size={15} /> {models?.length ? "Open training" : "Open models"}</button></div>
          </div>
          <div className="surface-divider" />
          <div className="inline-meta"><span><TerminalSquare size={14} /> CLI companion</span><code>{models?.length ? "axiom train plan 7 --method qlora" : "axiom model list"}</code></div>
        </Surface>

        <Surface className="module-surface" title="Platform map" eyebrow="WORKSPACE VIEWS" action={<button type="button" className="text-button" onClick={() => setPage("modules")}>All modules <ArrowRight size={15} /></button>}>
          <div className="module-list">{platformModules.map((module) => <ModuleRow key={module.page} module={module} onClick={() => setPage(module.page)} />)}</div>
        </Surface>

        <Surface title="About this workspace" eyebrow="LOCAL FIRST">
          <div className="empty-copy"><h3>{apiStatus === "online" ? "Backend is connected" : "Backend connection required"}</h3><p>Inventory and planning requests use the local AXIOM API. Plans are estimates; this interface does not start training or inference.</p><button type="button" className="text-button" onClick={() => setPage("logs")}>Open event log <ArrowRight size={15} /></button></div>
        </Surface>
      </div>
    </>
  );
}

function SummaryCard({ icon: Icon, label, value, detail, tone }: { icon: LucideIcon; label: string; value: string; detail: string; tone: Tone }) {
  return <div className="summary-card"><div className="summary-card-head"><span>{label}</span><Icon size={17} /></div><strong>{value}</strong><div><StatusPill tone={tone}>{detail}</StatusPill></div></div>;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; value >= 1024 && index < units.length; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(1)} ${unit}`;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function ModuleRow({ module, onClick }: { module: (typeof platformModules)[number]; onClick: () => void }) {
  const Icon = module.icon;
  return <button type="button" className="module-row" onClick={onClick}><span className="module-icon"><Icon size={17} /></span><span className="module-copy"><b>{module.title}</b><small>{module.description}</small></span><StatusPill tone={module.tone}>{module.state}</StatusPill><ChevronRight size={15} /></button>;
}

function FeaturePage({
  page,
  children,
  actions,
}: {
  page: Exclude<Page, "dashboard">;
  children: ReactNode;
  actions?: ReactNode;
}) {
  const meta = pageMeta[page];
  return <><PageHeader page={page} title={meta.label} description={meta.description} actions={actions} />{children}</>;
}

function Models({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [models, setModels] = useState<WorkspaceModel[]>([]);
  const [modelTotal, setModelTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [discoveryBusy, setDiscoveryBusy] = useState(false);
  const [discovery, setDiscovery] = useState<{ candidates: WorkspaceModelCandidate[]; truncated: boolean } | null>(null);
  const [inspection, setInspection] = useState<WorkspaceModelInspection | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [saving, setSaving] = useState(false);
  const [removing, setRemoving] = useState<WorkspaceModel | null>(null);
  const [removingBusy, setRemovingBusy] = useState(false);
  const [error, setError] = useState("");
  const [formOpen, setFormOpen] = useState(false);
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [format, setFormat] = useState("safetensors");
  const [parameters, setParameters] = useState("");
  const [architecture, setArchitecture] = useState("");
  const [license, setLicense] = useState("");
  const [quantization, setQuantization] = useState("");
  const [tags, setTags] = useState("");
  const importRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const page = await axiomApi.modelsPage(0, 25);
      setModels(page.models);
      setModelTotal(page.total);
      setHasMore(page.has_more);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Model registry unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const discoverLocalModels = async () => {
    setDiscoveryBusy(true);
    setInspection(null);
    try {
      const result = await axiomApi.discoverModels();
      setDiscovery(result);
      onNotify(result.candidates.length ? `Found ${result.candidates.length} local model folder${result.candidates.length === 1 ? "" : "s"}.` : "No local model folders found under models/.", result.candidates.length ? "success" : "neutral");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Local model discovery failed.", "warning");
    } finally {
      setDiscoveryBusy(false);
    }
  };

  const inspectCandidate = async (candidate: WorkspaceModelCandidate) => {
    try {
      const result = await axiomApi.inspectModel(candidate.path);
      setInspection(result.inspection);
      return result.inspection;
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Model inspection failed.", "warning");
      return null;
    }
  };

  const useCandidateMetadata = async (candidate: WorkspaceModelCandidate) => {
    const result = await inspectCandidate(candidate);
    if (!result) return;
    const fallbackName = candidate.path === "." ? "Local model" : candidate.path.split("/").filter(Boolean).at(-1) || "Local model";
    setName(fallbackName);
    setSource(candidate.path === "." ? "models" : `models/${candidate.path}`);
    setFormat(result.model_format.toLowerCase());
    setArchitecture(result.architecture || "");
    setParameters(result.parameter_count ? `${(result.parameter_count / 1_000_000_000).toFixed(1)}B` : "");
    setFormOpen(true);
  };

  const loadMore = async () => {
    setLoadingMore(true);
    try {
      const page = await axiomApi.modelsPage(models.length, 25);
      setModels((current) => [...current, ...page.models]);
      setModelTotal(page.total);
      setHasMore(page.has_more);
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "More models could not be loaded.", "warning");
    } finally {
      setLoadingMore(false);
    }
  };

  const addModel = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    try {
      await axiomApi.addModel({
        name,
        source,
        format,
        parameters: parameters.trim() || null,
        architecture: architecture.trim() || null,
        license: license.trim() || null,
        quantization: quantization.trim() || null,
        tags: [...new Set(tags.split(",").map((tag) => tag.trim()).filter(Boolean))],
        favorite: false,
      });
      setName("");
      setSource("");
      setParameters("");
      setArchitecture("");
      setLicense("");
      setQuantization("");
      setTags("");
      setFormOpen(false);
      onNotify("Model registered in the local AXIOM inventory.", "success");
      await refresh();
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Model registration failed.", "warning");
    } finally {
      setSaving(false);
    }
  };

  const toggleFavorite = async (model: WorkspaceModel) => {
    try {
      const result = await axiomApi.updateModel(model.id, { favorite: !model.favorite });
      setModels((current) => current.map((item) => item.id === model.id ? result.model : item));
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Favorite could not be updated.", "warning");
    }
  };

  const removeModel = async () => {
    if (!removing) return;
    setRemovingBusy(true);
    try {
      const result = await axiomApi.removeModel(removing.id);
      onNotify(`${result.removed.name} removed from registry. Local weights were not deleted.`, "success");
      setRemoving(null);
      await refresh();
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Model could not be removed.", "warning");
    } finally {
      setRemovingBusy(false);
    }
  };

  const exportModels = async () => {
    try {
      const payload = await axiomApi.exportModels();
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "axiom-model-registry.json";
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      onNotify("Model registry export downloaded.", "success");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Model registry export failed.", "warning");
    }
  };

  const importModels = async (event: ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    try {
      if (file.size > 4 * 1024 * 1024) throw new Error("Import files must be 4 MB or smaller.");
      const payload = JSON.parse(await file.text()) as { schema_version?: unknown; models?: unknown };
      if (payload.schema_version !== 1 || !Array.isArray(payload.models)) {
        throw new Error("Choose an AXIOM model registry export with schema_version 1.");
      }
      const result = await axiomApi.importModels({ schema_version: 1, models: payload.models as WorkspaceModel[] });
      onNotify(`Imported ${result.imported} model records.`, "success");
      await refresh();
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Model registry import failed.", "warning");
    } finally {
      input.value = "";
    }
  };

  return <FeaturePage page="models" actions={<><button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}><RefreshCcw size={16} /> Refresh</button><button type="button" className="button secondary-button" onClick={() => void discoverLocalModels()} disabled={discoveryBusy}><Search size={15} /> {discoveryBusy ? "Discovering…" : "Discover local"}</button><button type="button" className="button secondary-button" onClick={() => void exportModels()}><ArrowDownToLine size={15} /> Export</button><button type="button" className="button secondary-button" onClick={() => importRef.current?.click()}><Upload size={15} /> Import</button><input ref={importRef} type="file" accept="application/json,.json" hidden onChange={(event) => void importModels(event)} /><button type="button" className="button primary-button" onClick={() => setFormOpen((open) => !open)}><Plus size={16} /> Register model</button></>}>
    <div className="info-grid"><InfoCard icon={BrainCircuit} label="Registry" value={loading ? "—" : String(modelTotal)} detail={loading ? "Loading inventory" : `${models.filter((model) => model.favorite).length} favorites loaded`} tone={modelTotal ? "success" : "neutral"} /><InfoCard icon={HardDrive} label="Storage" value="Local" detail="Registry metadata only" tone="neutral" /><InfoCard icon={SlidersHorizontal} label="Inventory" value={loading ? "—" : String(models.length)} detail={hasMore ? `Showing ${models.length} of ${modelTotal}` : "All registry entries shown"} tone="info" /></div>
    {formOpen && <Surface title="Register model metadata" eyebrow="LOCAL REGISTRY">
      <form className="integration-form" onSubmit={(event) => void addModel(event)}>
        <label className="field"><span className="field-label-row"><span>Display name</span></span><input value={name} onChange={(event) => setName(event.target.value)} maxLength={160} required placeholder="Qwen 3 8B" /></label>
        <label className="field"><span className="field-label-row"><span>Source or local path</span></span><input value={source} onChange={(event) => setSource(event.target.value)} maxLength={512} required placeholder="./models/qwen3-8b" /></label>
        <label className="field"><span className="field-label-row"><span>Format</span></span><input value={format} onChange={(event) => setFormat(event.target.value)} maxLength={64} required placeholder="safetensors, gguf, ..." /></label>
        <label className="field"><span className="field-label-row"><span>Architecture</span></span><input value={architecture} onChange={(event) => setArchitecture(event.target.value)} maxLength={160} placeholder="Qwen3ForCausalLM" /></label>
        <label className="field"><span className="field-label-row"><span>Parameters</span></span><input value={parameters} onChange={(event) => setParameters(event.target.value)} maxLength={80} placeholder="8B" /></label>
        <label className="field"><span className="field-label-row"><span>Quantization</span></span><input value={quantization} onChange={(event) => setQuantization(event.target.value)} maxLength={80} placeholder="Q4_K_M" /></label>
        <label className="field"><span className="field-label-row"><span>License</span></span><input value={license} onChange={(event) => setLicense(event.target.value)} maxLength={120} placeholder="Apache-2.0" /></label>
        <label className="field"><span className="field-label-row"><span>Tags, comma-separated</span></span><input value={tags} onChange={(event) => setTags(event.target.value)} maxLength={1200} placeholder="chat, local, favorite" /></label>
        <div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => setFormOpen(false)}>Cancel</button><button type="submit" className="button primary-button" disabled={saving || !name.trim() || !source.trim() || !format.trim()}>{saving ? "Registering…" : "Save to registry"}</button></div>
      </form>
      <p className="quiet-note">This saves metadata only. It does not download, move, or inspect model files. Registry import accepts AXIOM schema version 1 and never overwrites existing entries.</p>
    </Surface>}
    {discovery && <Surface title="Local model folders" eyebrow="SAFE DISCOVERY">
      <p className="quiet-note">Scans at most 5,000 folders, four levels deep, and reports at most 100 candidates. Symlinked folders and files are skipped. Discovery is limited to models/.</p>
      {discovery.truncated && <Callout tone="warning" title="Discovery was bounded">Some folders may not be listed. Narrow the models/ tree and retry.</Callout>}
      {discovery.candidates.length ? <div className="module-list">{discovery.candidates.map((candidate) => <div className="module-row" key={candidate.path}>
        <span className="module-icon"><HardDrive size={17} /></span><span className="module-copy"><b>{candidate.path === "." ? "models/" : `models/${candidate.path}`}</b><small>{candidate.format} · {candidate.weight_file_count} weight files · {formatBytes(candidate.weight_size_bytes)}{candidate.has_config ? " · config.json" : " · no config"}</small></span>
        <button type="button" className="button secondary-button" onClick={() => void inspectCandidate(candidate)}>Inspect</button><button type="button" className="button secondary-button" onClick={() => void useCandidateMetadata(candidate)}>Use metadata</button>
      </div>)}</div> : <EmptyState icon={Search} title="No candidate folders found" description="Create a local model folder under models/ with config.json or supported weight files." />}
    </Surface>}
    {inspection && <Surface title="Model inspection summary" eyebrow="LOCAL FILE METADATA" action={<button type="button" className="icon-text-button" onClick={() => setInspection(null)}>Dismiss</button>}>
      <p className="quiet-note">{inspection.path} · {inspection.status}</p>
      <div className="info-grid"><InfoCard icon={BrainCircuit} label="Architecture" value={inspection.architecture || inspection.model_type || "Unknown"} detail={inspection.capabilities.join(" · ")} tone="neutral" /><InfoCard icon={HardDrive} label="Format / weights" value={inspection.model_format} detail={`${formatBytes(inspection.weight_size_bytes)} · ${inspection.has_tokenizer ? "Tokenizer found" : "No tokenizer detected"}`} tone={inspection.weight_size_bytes ? "success" : "warning"} /><InfoCard icon={Cpu} label="Estimated VRAM" value={inspection.estimated_vram_gb === null ? "Unknown" : `${inspection.estimated_vram_gb} GB`} detail={inspection.parameter_count ? `${inspection.parameter_count.toLocaleString()} parameters · estimate only` : "No parameter count available"} tone="info" /></div>
    </Surface>}
    <Surface title="Model registry" eyebrow="LOCAL INVENTORY">
      {error ? <InlineAlert tone="warning" role="alert">{error}</InlineAlert> : null}
      {loading ? <div className="loading-state" role="status">Loading model registry…</div> : models.length ? <><div className="module-list">{models.map((model) => <div className="module-row model-registry-row" key={model.id}><span className="module-icon"><BrainCircuit size={17} /></span><span className="module-copy"><b>{model.name}</b><small>{model.source} · {model.format}{model.parameters ? ` · ${model.parameters}` : ""}{model.quantization ? ` · ${model.quantization}` : ""}{model.architecture ? ` · ${model.architecture}` : ""}{model.license ? ` · ${model.license}` : ""}{model.tags?.length ? ` · ${model.tags.join(", ")}` : ""}</small></span><button type="button" className="icon-button" aria-label={`${model.favorite ? "Remove" : "Add"} ${model.name} ${model.favorite ? "favorite" : "to favorites"}`} title={model.favorite ? "Remove favorite" : "Add favorite"} onClick={() => void toggleFavorite(model)}><Star size={15} fill={model.favorite ? "currentColor" : "none"} /></button><button type="button" className="icon-button danger-icon-button" aria-label={`Remove ${model.name} from registry`} title="Remove registry entry" onClick={() => setRemoving(model)}><Trash2 size={15} /></button></div>)}</div>{hasMore && <div className="pagination-actions"><button type="button" className="button secondary-button" onClick={() => void loadMore()} disabled={loadingMore}>{loadingMore ? "Loading…" : "Load more models"}</button></div>}</> : <EmptyState icon={BrainCircuit} title="Registry is empty" description="Register model metadata here or use the CLI. Model files remain where they are." action={<CliReference command="axiom model add my-model local safetensors" />} />}
    </Surface>
    {removing && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !removingBusy) setRemoving(null); }}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="remove-model-title"><div className="dialog-head"><div><p className="eyebrow">CONFIRM REGISTRY CHANGE</p><h2 id="remove-model-title">Remove {removing.name}?</h2></div><button type="button" className="icon-button" aria-label="Close confirmation" onClick={() => setRemoving(null)} disabled={removingBusy}><X size={16} /></button></div><p className="quiet-note">This removes only its AXIOM registry metadata. Local model weights will not be deleted.</p><div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => setRemoving(null)} disabled={removingBusy}>Cancel</button><button type="button" className="button danger-button" onClick={() => void removeModel()} disabled={removingBusy}>{removingBusy ? "Removing…" : "Remove registry entry"}</button></div></section></div>}
  </FeaturePage>;
}

const datasetScanPresets = {
  quick: {
    label: "Quick",
    description: "10,000 rows · 8 MB · 256 KB per line",
    limits: { max_bytes: 8 * 1024 * 1024, max_rows: 10_000, max_line_bytes: 256 * 1024 },
  },
  standard: {
    label: "Standard",
    description: "100,000 rows · 64 MB · 1 MB per line",
    limits: { max_bytes: 64 * 1024 * 1024, max_rows: 100_000, max_line_bytes: 1024 * 1024 },
  },
  deep: {
    label: "Deep",
    description: "500,000 rows · 256 MB · 8 MB per line",
    limits: { max_bytes: 256 * 1024 * 1024, max_rows: 500_000, max_line_bytes: 8 * 1024 * 1024 },
  },
} as const;

function Datasets({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [datasets, setDatasets] = useState<WorkspaceDataset[]>([]);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [parentManifestSha256, setParentManifestSha256] = useState("");
  const [cleanPreview, setCleanPreview] = useState<DatasetCleaningPreview | null>(null);
  const [comparePaths, setComparePaths] = useState<string[]>([]);
  const [requiredFields, setRequiredFields] = useState("");
  const [expectedTypeRules, setExpectedTypeRules] = useState("");
  const [duplicateKeyFields, setDuplicateKeyFields] = useState("");
  const [duplicateRateThreshold, setDuplicateRateThreshold] = useState("");
  const [textFields, setTextFields] = useState("");
  const [sensitiveFieldRules, setSensitiveFieldRules] = useState("");
  const [ignoredSensitiveFields, setIgnoredSensitiveFields] = useState("");
  const [scanPii, setScanPii] = useState(false);
  const [qualityReport, setQualityReport] = useState<DatasetQualityResult | null>(null);
  const [qualityBusy, setQualityBusy] = useState(false);
  const [scanPreset, setScanPreset] = useState<keyof typeof datasetScanPresets>("standard");
  const [profileBusy, setProfileBusy] = useState<string | null>(null);
  const [manifestBusy, setManifestBusy] = useState<string | null>(null);
  const [cleanPreviewBusy, setCleanPreviewBusy] = useState<string | null>(null);
  const [inventoryBusy, setInventoryBusy] = useState(true);
  const [error, setError] = useState("");

  const refresh = async () => {
    setInventoryBusy(true);
    setError("");
    try {
      setDatasets(await axiomApi.datasets());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Dataset inventory unavailable.");
    } finally {
      setInventoryBusy(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const parseProfileOptions = () => {
    const parseFields = (value: string) => [...new Set(value.split(",").map((field) => field.trim()).filter(Boolean))];
    const duplicateFields = parseFields(duplicateKeyFields);
    const selectedTextFields = parseFields(textFields);
    const rules = parseFields(sensitiveFieldRules);
    const exclusions = parseFields(ignoredSensitiveFields);
    if ([...duplicateFields, ...selectedTextFields, ...rules, ...exclusions].some((field) => field.length > 160) || [duplicateFields, selectedTextFields, rules, exclusions].some((fields) => fields.length > 100)) {
      onNotify("Use at most 100 profile fields per group, each 160 characters or fewer.", "warning");
      return null;
    }
    const threshold = duplicateRateThreshold.trim() === "" ? null : Number(duplicateRateThreshold);
    if (threshold !== null && (!Number.isFinite(threshold) || threshold < 0 || threshold > 1)) {
      onNotify("Duplicate-rate threshold must be between 0 and 1.", "warning");
      return null;
    }
    return {
      duplicate_key_fields: duplicateFields,
      duplicate_rate_threshold: threshold,
      text_fields: selectedTextFields,
      sensitive_field_rules: rules,
      ignored_sensitive_fields: exclusions,
      scan_pii: scanPii,
    };
  };

  const runProfile = async (path: string) => {
    const options = parseProfileOptions();
    if (!options) return;
    setProfileBusy(path);
    setProfile(null);
    try {
      const report = await axiomApi.profileDataset(path, datasetScanPresets[scanPreset].limits, options);
      setProfile(report);
      onNotify(`Profile ready: ${report.valid_rows.toLocaleString()} valid rows.`, "success");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Dataset profiling failed.", "warning");
    } finally {
      setProfileBusy(null);
    }
  };

  const runManifest = async (path: string) => {
    const parentHash = parentManifestSha256.trim();
    if (parentHash && !/^[0-9a-f]{64}$/i.test(parentHash)) {
      onNotify("Parent manifest must be a 64-character SHA-256 hex digest.", "warning");
      return;
    }
    setManifestBusy(path);
    setManifest(null);
    try {
      setManifest(await axiomApi.datasetManifest(path, datasetScanPresets[scanPreset].limits.max_bytes, parentHash || undefined));
      onNotify("Dataset provenance manifest ready.", "success");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Dataset manifest failed.", "warning");
    } finally {
      setManifestBusy(null);
    }
  };

  const runCleaningPreview = async (path: string) => {
    setCleanPreviewBusy(path);
    setCleanPreview(null);
    try {
      setCleanPreview(await axiomApi.previewDatasetCleaning(path, datasetScanPresets[scanPreset].limits));
      onNotify("Cleaning preview ready; no file was written.", "success");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Dataset cleaning preview failed.", "warning");
    } finally {
      setCleanPreviewBusy(null);
    }
  };

  const runQualityChecks = async () => {
    const options = parseProfileOptions();
    if (!options) return;
    const expectedTypes: Record<string, string> = {};
    const supportedTypes = new Set(["string", "integer", "number", "boolean", "object", "array", "null"]);
    for (const rule of expectedTypeRules.split(/\r?\n/).map((value) => value.trim()).filter(Boolean)) {
      const separator = rule.lastIndexOf("=");
      const field = rule.slice(0, separator).trim();
      const type = rule.slice(separator + 1).trim();
      if (separator < 1 || !field || !supportedTypes.has(type)) {
        onNotify("Use one field=type rule per line with a supported JSON type.", "warning");
        return;
      }
      expectedTypes[field] = type;
    }
    const fields = [...new Set(requiredFields.split(",").map((value) => value.trim()).filter(Boolean))];
    if (fields.length > 100 || Object.keys(expectedTypes).length > 100) {
      onNotify("Use at most 100 required fields and 100 type rules.", "warning");
      return;
    }
    setQualityBusy(true);
    try {
      const report = await axiomApi.datasetQuality({
        paths: comparePaths,
        required_fields: fields,
        expected_types: expectedTypes,
        ...options,
        limits: datasetScanPresets[scanPreset].limits,
      });
      setQualityReport(report);
      const passed = Object.values(report.checks).every((check) => check.passed);
      onNotify(passed ? "Dataset comparison and checks passed." : "Dataset report ready; review the schema differences.", passed ? "success" : "warning");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Dataset quality report failed.", "warning");
    } finally {
      setQualityBusy(false);
    }
  };

  const exportQualityReport = () => {
    if (!qualityReport) return;
    const blob = new Blob([JSON.stringify({ generated_at: new Date().toISOString(), ...qualityReport }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "axiom-dataset-quality-report.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const exportManifest = () => {
    if (!manifest) return;
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${manifest.file_name}.manifest.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const exportCleaningPreview = () => {
    if (!cleanPreview) return;
    const blob = new Blob([JSON.stringify(cleanPreview, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${cleanPreview.file_name}.cleaning-preview.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  const clearReports = () => {
    setProfile(null);
    setQualityReport(null);
    setManifest(null);
    setCleanPreview(null);
  };

  return <FeaturePage page="datasets" actions={<div className="dataset-page-actions"><button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={inventoryBusy}><RefreshCcw size={16} /> {inventoryBusy ? "Refreshing" : "Refresh inventory"}</button><button type="button" className="button secondary-button" onClick={clearReports} disabled={(!profile && !qualityReport && !manifest && !cleanPreview) || profileBusy !== null || qualityBusy || manifestBusy !== null || cleanPreviewBusy !== null}><Trash2 size={15} /> Clear reports</button></div>}>
    <div className="info-grid">
      <InfoCard icon={Database} label="Datasets" value={inventoryBusy ? "—" : String(datasets.length)} detail={inventoryBusy ? "Loading inventory" : "JSONL files under data/"} tone={datasets.length ? "success" : "neutral"} />
      <InfoCard icon={CircleAlert} label="Profiler" value="Local" detail="Bounded structural scan" tone="success" />
      <InfoCard icon={BarChart3} label="Privacy review" value={profile?.privacy_review_recommended ? "Recommended" : "On demand"} detail="Field names only; values stay private" tone={profile?.privacy_review_recommended ? "warning" : "neutral"} />
    </div>
    <Surface title="Dataset workspace" eyebrow="INSPECTION QUEUE">
      {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
      <div className="dataset-scan-controls">
        <label htmlFor="dataset-scan-preset">
          <span>Scan budget</span>
          <select id="dataset-scan-preset" value={scanPreset} onChange={(event) => setScanPreset(event.target.value as keyof typeof datasetScanPresets)}>
            {Object.entries(datasetScanPresets).map(([key, preset]) => <option key={key} value={key}>{preset.label}</option>)}
          </select>
        </label>
        <p className="quiet-note">{datasetScanPresets[scanPreset].description}. The API enforces fixed maximums; values are never returned.</p>
      </div>
      <label className="dataset-manifest-parent"><span>Optional parent manifest SHA-256 · link this dataset to an earlier source</span><input value={parentManifestSha256} onChange={(event) => setParentManifestSha256(event.target.value)} maxLength={64} placeholder="Paste a previous manifest digest (leave blank for source)" /></label>
      {inventoryBusy ? <div className="loading-state" role="status">Scanning the local data directory…</div> : datasets.length ? <div className="module-list">{datasets.map((dataset) => <div className="module-row" key={dataset.path}>
        <span className="module-icon"><Database size={17} /></span>
        <span className="module-copy"><b>{dataset.path}</b><small>JSONL · {formatBytes(dataset.size_bytes)} · local workspace</small></span>
        <div className="module-row-actions"><button type="button" className="button secondary-button" disabled={profileBusy !== null} onClick={() => void runProfile(dataset.path)}>{profileBusy === dataset.path ? "Profiling…" : "Profile"}</button><button type="button" className="button secondary-button" disabled={manifestBusy !== null} onClick={() => void runManifest(dataset.path)}>{manifestBusy === dataset.path ? "Hashing…" : "Manifest"}</button><button type="button" className="button secondary-button" disabled={cleanPreviewBusy !== null} onClick={() => void runCleaningPreview(dataset.path)}>{cleanPreviewBusy === dataset.path ? "Previewing…" : "Preview clean"}</button></div>
      </div>)}</div> : <EmptyState icon={Database} title="No JSONL datasets found" description="Place a JSONL file under the workspace data/ directory, then refresh. You can inspect any path directly with the CLI." action={<CliReference command="axiom dataset profile ./data/train.jsonl" />} />}
    </Surface>
    {datasets.length > 0 && <Surface title="Compare and validate" eyebrow="DATASET QUALITY">
      <p className="quiet-note">Select two or three JSONL files. The report compares field presence and observed types; checks never return row values.</p>
      <div className="dataset-quality-selection">
        {datasets.map((dataset) => <label className="dataset-quality-choice" key={dataset.path}>
          <input
            type="checkbox"
            checked={comparePaths.includes(dataset.path)}
            disabled={qualityBusy || (!comparePaths.includes(dataset.path) && comparePaths.length >= 3)}
            onChange={() => setComparePaths((current) => current.includes(dataset.path) ? current.filter((path) => path !== dataset.path) : [...current, dataset.path])}
          />
          <span><b>{dataset.path}</b><small>{formatBytes(dataset.size_bytes)}</small></span>
        </label>)}
      </div>
      <div className="dataset-quality-form">
        <label className="field"><span className="field-label-row"><span>Required fields</span></span><input value={requiredFields} onChange={(event) => setRequiredFields(event.target.value)} maxLength={2000} placeholder="prompt, completion, label" /></label>
        <label className="field"><span className="field-label-row"><span>Expected types, one per line</span></span><textarea value={expectedTypeRules} onChange={(event) => setExpectedTypeRules(event.target.value)} rows={3} maxLength={4000} placeholder={'count=integer\nscore=number\ntext=string'} /></label>
        <label className="field"><span className="field-label-row"><span>Duplicate keys</span></span><input value={duplicateKeyFields} onChange={(event) => setDuplicateKeyFields(event.target.value)} maxLength={2000} placeholder="id, source_id" /></label>
        <label className="field"><span className="field-label-row"><span>Duplicate-rate threshold (0–1)</span></span><input inputMode="decimal" value={duplicateRateThreshold} onChange={(event) => setDuplicateRateThreshold(event.target.value)} maxLength={8} placeholder="0.05" /></label>
        <label className="field"><span className="field-label-row"><span>Text fields for token-length summary</span></span><input value={textFields} onChange={(event) => setTextFields(event.target.value)} maxLength={2000} placeholder="prompt, completion" /></label>
        <label className="field"><span className="field-label-row"><span>Additional sensitive-name rules</span></span><input value={sensitiveFieldRules} onChange={(event) => setSensitiveFieldRules(event.target.value)} maxLength={2000} placeholder="customer_ref, employee_id" /></label>
        <label className="field"><span className="field-label-row"><span>Ignore exact names (false-positive control)</span></span><input value={ignoredSensitiveFields} onChange={(event) => setIgnoredSensitiveFields(event.target.value)} maxLength={2000} placeholder="email_template" /></label>
      </div>
      <label className="dataset-scan-optin"><input type="checkbox" checked={scanPii} onChange={(event) => setScanPii(event.target.checked)} /><span><b>Explicitly scan local values for likely PII</b><small>Off by default. Scans bounded text for email, phone-like, and payment-card patterns; matched values are never returned.</small></span></label>
      <div className="integration-form-actions">
        <button type="button" className="button primary-button" disabled={qualityBusy || comparePaths.length < 2} onClick={() => void runQualityChecks()}>{qualityBusy ? "Comparing…" : `Compare ${comparePaths.length || "datasets"}`}</button>
        {qualityReport && <button type="button" className="button secondary-button" onClick={exportQualityReport}><ArrowDownToLine size={15} /> Export JSON report</button>}
      </div>
    </Surface>}
    {manifest && <Surface title={`Manifest: ${manifest.file_name}`} eyebrow="DATASET PROVENANCE" action={<button type="button" className="button secondary-button" onClick={exportManifest}><ArrowDownToLine size={15} /> Export JSON</button>}>
      <div className="info-grid">
        <InfoCard icon={ShieldCheck} label="Hash status" value={manifest.status.replaceAll("_", " ")} detail={manifest.full_content_hashed ? "Full content SHA-256" : `${formatBytes(manifest.bytes_hashed)} hashed of ${formatBytes(manifest.file_size_bytes)}`} tone={manifest.full_content_hashed ? "success" : "warning"} />
        <InfoCard icon={HardDrive} label="File size" value={formatBytes(manifest.file_size_bytes)} detail={`${manifest.path} · ${manifest.lineage.relation}`} tone="neutral" />
        <InfoCard icon={Clock3} label="Generated" value={new Date(manifest.generated_at).toLocaleString()} detail={manifest.modified_at ? `Modified ${new Date(manifest.modified_at).toLocaleString()}` : "Modification time unavailable"} tone="neutral" />
      </div>
      <div className="dataset-quality-check passed"><b>{manifest.full_content_hashed ? "SHA-256" : manifest.status === "partial" ? "Partial SHA-256" : "Hash unavailable"}</b><small className="manifest-digest">{manifest.sha256 ?? manifest.partial_sha256 ?? "The source changed while it was being read; retry to create a stable checksum."}</small>{manifest.lineage.parent_manifest_sha256 && <small className="manifest-digest">Parent manifest SHA-256: {manifest.lineage.parent_manifest_sha256}</small>}<small>Manifest contains file metadata and checksums only, never dataset values.</small></div>
    </Surface>}
    {cleanPreview && <Surface title={`Cleaning preview: ${cleanPreview.file_name}`} eyebrow="NO-WRITE PREVIEW" action={<button type="button" className="button secondary-button" onClick={exportCleaningPreview}><ArrowDownToLine size={15} /> Export JSON</button>}>
      <Callout tone="success" title="Preview only">No output file was created or changed. The counts simulate the existing JSONL cleaner for the scanned portion; use `axiom dataset clean` only after reviewing.</Callout>
      <div className="info-grid">
        <InfoCard icon={Database} label="Kept rows" value={cleanPreview.kept_rows.toLocaleString()} detail={`${cleanPreview.removed_rows.toLocaleString()} rows removed by current rules`} tone="success" />
        <InfoCard icon={CircleAlert} label="Rewritten rows" value={cleanPreview.rewritten_rows.toLocaleString()} detail="JSON formatting or line-ending normalization" tone={cleanPreview.rewritten_rows ? "warning" : "neutral"} />
        <InfoCard icon={HardDrive} label="Estimated output" value={formatBytes(cleanPreview.estimated_output_bytes)} detail={`${formatBytes(cleanPreview.bytes_scanned)} scanned${cleanPreview.truncated ? " · partial preview" : " · complete scan"}`} tone="neutral" />
      </div>
      <div className="dataset-quality-checks">{cleanPreview.changes.filter((change) => change.rows > 0).map((change) => <div key={change.operation} className="dataset-quality-check passed"><b>{change.operation.replaceAll("_", " ")}</b><small>{change.rows.toLocaleString()} rows</small></div>)}</div>
      {cleanPreview.warnings.map((warning) => <p className="quiet-note" key={warning}>{warning}</p>)}
    </Surface>}
    {qualityReport && <Surface title="Schema comparison report" eyebrow="LOCAL QUALITY REPORT">
      <div className="info-grid">
        <InfoCard icon={Database} label="Datasets" value={String(qualityReport.comparison.dataset_count)} detail="Compared with the same scan budget" tone="neutral" />
        <InfoCard icon={CircleAlert} label="Schema differences" value={qualityReport.comparison.has_differences ? "Found" : "None"} detail="Field presence or observed type" tone={qualityReport.comparison.has_differences ? "warning" : "success"} />
        <InfoCard icon={CircleCheck} label="Validation checks" value={`${Object.values(qualityReport.checks).filter((check) => check.passed).length}/${qualityReport.comparison.dataset_count}`} detail="Required fields and expected types" tone={Object.values(qualityReport.checks).every((check) => check.passed) ? "success" : "warning"} />
      </div>
      <p className="quiet-note">Per-file limits: {qualityReport.limits.max_rows.toLocaleString()} rows · {formatBytes(qualityReport.limits.max_bytes)} · {formatBytes(qualityReport.limits.max_line_bytes)} per line. Truncated profiles may omit later fields.</p>
      <div className="module-list">{qualityReport.comparison.fields.map((field) => <div className="module-row" key={field.name}>
        <span className="module-icon"><BarChart3 size={17} /></span>
        <span className="module-copy"><b>{field.name}{field.possibly_sensitive ? " · review" : ""}</b><small>{field.missing_from.length ? `Missing from: ${field.missing_from.join(", ")}` : `Present in all ${qualityReport.comparison.dataset_count} datasets`} · {Object.entries(field.types_by_dataset).map(([path, types]) => `${path}: ${types.join("/")}`).join(" · ")}</small></span>
        <StatusPill tone={field.missing_from.length || field.type_mismatch ? "warning" : "success"}>{field.missing_from.length || field.type_mismatch ? "REVIEW" : "MATCH"}</StatusPill>
      </div>)}</div>
      <div className="dataset-quality-checks">{Object.entries(qualityReport.checks).map(([path, check]) => <div key={path} className={check.passed ? "dataset-quality-check passed" : "dataset-quality-check failed"}>
        <b>{path} · {check.passed ? "checks passed" : "needs review"}</b>
        {check.missing_fields.length > 0 && <small>Missing required fields: {check.missing_fields.join(", ")}</small>}
        {check.type_mismatches.map((mismatch) => <small key={mismatch.field}>{mismatch.field}: expected {mismatch.expected}, observed {mismatch.observed.join(", ") || "no values"}</small>)}
      </div>)}</div>
      <div className="dataset-quality-checks">{qualityReport.datasets.map(({ path, profile: item }) => <div key={`diagnostics-${path}`} className="dataset-quality-check passed">
        <b>{path} · scan diagnostics</b>
        <small>{item.diagnostics.newline_style} line endings · {item.diagnostics.encoding_errors} UTF-8 errors · {item.diagnostics.malformed_json_rows} malformed JSON · {item.diagnostics.non_object_json_rows} non-object rows{item.truncated ? ` · partial due to ${item.truncation_reasons.join(", ") || "scan limit"}` : ""}</small>
        {item.duplicate_key_summary.fields.length > 0 && <small>Duplicate keys ({item.duplicate_key_summary.fields.join(", ")}): {item.duplicate_key_summary.duplicate_rows}/{item.duplicate_key_summary.eligible_rows} · {(item.duplicate_key_summary.rate * 100).toFixed(2)}%{item.duplicate_key_summary.threshold_exceeded ? " · THRESHOLD EXCEEDED" : ""}</small>}
        {Object.entries(item.text_token_lengths).map(([field, stats]) => <small key={field}>{field} whitespace-token count: {stats.observations} values · {Object.entries(stats.histogram).map(([bucket, count]) => `${bucket}: ${count}`).join(", ") || "no string values"}</small>)}
        {item.pii_scan.opted_in && <small>Local PII scan: {Object.entries(item.pii_scan.matches_by_type).map(([kind, count]) => `${kind.replaceAll("_", " ")}: ${count}`).join(" · ")} · fields: {Object.values(item.pii_scan.fields_by_type).flat().join(", ") || "none"} · match values withheld</small>}
      </div>)}</div>
    </Surface>}
    {profile && <Surface title={`Profile: ${profile.file_name}`} eyebrow="JSONL STRUCTURE REPORT" action={<button type="button" className="icon-text-button" onClick={() => setProfile(null)}>Clear report</button>}>
      <div className="info-grid">
        <InfoCard icon={Database} label="Valid rows" value={profile.valid_rows.toLocaleString()} detail="JSON object rows" tone="success" />
        <InfoCard icon={CircleAlert} label="Invalid rows" value={profile.invalid_rows.toLocaleString()} detail={`${profile.empty_rows} blank · ${profile.overlong_rows} oversized`} tone={profile.invalid_rows ? "warning" : "success"} />
        <InfoCard icon={RefreshCcw} label="Duplicates" value={profile.duplicate_rows.toLocaleString()} detail="Repeated valid rows" tone={profile.duplicate_rows ? "warning" : "neutral"} />
      </div>
      <p className="quiet-note">Scanned {formatBytes(profile.bytes_scanned)} of {formatBytes(profile.file_size_bytes)} across {profile.lines_scanned.toLocaleString()} lines{profile.truncated ? ` · Partial report: ${profile.truncation_reasons.join(", ") || "scan limit reached"}` : " · Complete scan"}. Limits: {profile.limits.max_rows.toLocaleString()} rows · {formatBytes(profile.limits.max_bytes)} · {formatBytes(profile.limits.max_line_bytes)} per line.</p>
      <div className="dataset-quality-check passed"><b>Scan diagnostics</b><small>{profile.diagnostics.newline_style} line endings · {profile.diagnostics.encoding_errors} UTF-8 errors · {profile.diagnostics.malformed_json_rows} malformed JSON · {profile.diagnostics.non_object_json_rows} non-object rows</small>{profile.duplicate_key_summary.fields.length > 0 && <small>Duplicate keys ({profile.duplicate_key_summary.fields.join(", ")}): {profile.duplicate_key_summary.duplicate_rows}/{profile.duplicate_key_summary.eligible_rows} · {(profile.duplicate_key_summary.rate * 100).toFixed(2)}%{profile.duplicate_key_summary.threshold_exceeded ? " · THRESHOLD EXCEEDED" : ""}</small>}{Object.entries(profile.text_token_lengths).map(([field, stats]) => <small key={field}>{field} whitespace-token count: {stats.observations} values · {Object.entries(stats.histogram).map(([bucket, count]) => `${bucket}: ${count}`).join(", ") || "no string values"}</small>)}<small>Sensitive-name rules: {profile.sensitive_field_rules.length ? profile.sensitive_field_rules.join(", ") : "built-in defaults"} · exclusions: {profile.ignored_sensitive_fields.join(", ") || "none"}</small></div>
      {profile.privacy_review_recommended && <Callout tone="warning" title="Privacy review recommended">Potentially sensitive field names: {profile.possibly_sensitive_fields.join(", ")}. Raw dataset values are not returned.</Callout>}
      {profile.pii_scan.opted_in && <Callout tone={Object.values(profile.pii_scan.matches_by_type).some((count) => count > 0) ? "warning" : "success"} title={Object.values(profile.pii_scan.matches_by_type).some((count) => count > 0) ? "Likely PII patterns detected" : "No likely PII patterns detected"}>{Object.entries(profile.pii_scan.matches_by_type).map(([kind, count]) => `${kind.replaceAll("_", " ")}: ${count}`).join(" · ")} · fields: {Object.entries(profile.pii_scan.fields_by_type).flatMap(([kind, fields]) => fields.map((field) => `${field} (${kind.replaceAll("_", " ")})`)).join(", ") || "none"}. Values scanned: {profile.pii_scan.values_scanned.toLocaleString()}; {profile.pii_scan.long_values_skipped} long values skipped; matched text is never shown. This heuristic scan is not a guarantee of detection.</Callout>}
      {profile.warnings.map((warning) => <p className="quiet-note" key={warning}>{warning}</p>)}
      <div className="module-list">{profile.fields.map((field) => <div className="module-row" key={field.name}>
        <span className="module-icon"><BarChart3 size={17} /></span>
        <span className="module-copy"><b>{field.name}{field.possibly_sensitive ? " · review" : ""}</b><small>{field.occurrences} values · {field.null_count} null · types: {Object.entries(field.types).map(([kind, count]) => `${kind} ${count}`).join(", ")}</small></span>
        <small>{(field.null_rate * 100).toFixed(1)}% null</small>
      </div>)}</div>
    </Surface>}
  </FeaturePage>;
}

function Training({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [parameterBillions, setParameterBillions] = useState("7");
  const [method, setMethod] = useState("auto");
  const [plan, setPlan] = useState<TrainingPlanResult | null>(null);
  const [hardware, setHardware] = useState<RuntimeSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [savingExperiment, setSavingExperiment] = useState(false);
  const [savedExperimentId, setSavedExperimentId] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    axiomApi.runtime().then(setHardware).catch(() => setHardware(null));
  }, []);

  const createPlan = async () => {
    const size = Number(parameterBillions);
    if (!Number.isFinite(size) || size <= 0 || size > 1000) {
      setError("Enter a model size between 0 and 1000 billion parameters.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await axiomApi.trainingPlan(size, method);
      setPlan(result);
      onNotify(`Training plan ready: ${result.method}, ${result.fits_hardware ? "estimated to fit" : "review the hardware estimate"}.`, result.fits_hardware ? "success" : "warning");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Training plan unavailable.");
    } finally {
      setBusy(false);
    }
  };

  const saveExperiment = async () => {
    if (!plan) return;
    setSavingExperiment(true);
    setError("");
    try {
      const result = await axiomApi.createExperiment({
        name: `${parameterBillions}B ${plan.method} plan`,
        parameter_billions: Number(parameterBillions),
        method,
        plan_snapshot: plan,
      });
      setSavedExperimentId(result.experiment.id);
      onNotify("Plan saved to persistent experiment history. No job was started.", "success");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment could not be saved.");
    } finally {
      setSavingExperiment(false);
    }
  };

  return <FeaturePage page="training" actions={<button type="button" className="button primary-button" onClick={() => void createPlan()} disabled={busy}><Play size={16} /> {busy ? "Building plan…" : "Build plan"}</button>}>
    <div className="info-grid">
      <InfoCard icon={Zap} label="Method" value={plan?.method.toUpperCase() || method.toUpperCase()} detail={plan ? plan.precision : "Choose a planning method"} tone={plan?.fits_hardware ? "success" : "neutral"} />
      <InfoCard icon={Cpu} label="Detected hardware" value={hardware?.hardware.gpu_name || "CPU only"} detail={hardware ? `${hardware.hardware.ram_gb.toFixed(1)} GB RAM · ${hardware.hardware.vram_gb ?? 0} GB VRAM` : "Hardware unavailable"} tone={hardware?.hardware.cuda_available ? "success" : "warning"} />
      <InfoCard icon={TimerReset} label="Execution" value="Plan only" detail="No training job is started" tone="neutral" />
    </div>
    <div className="two-column-grid">
      <Surface title="Plan inputs" eyebrow="HARDWARE-AWARE ESTIMATE">
        <label className="field"><span className="field-label-row"><span>Model size (billions of parameters)</span></span><input type="number" min="0.1" max="1000" step="0.1" value={parameterBillions} onChange={(event) => { setParameterBillions(event.target.value); setPlan(null); setSavedExperimentId(""); }} /></label>
        <label className="field"><span className="field-label-row"><span>Training method</span></span><select value={method} onChange={(event) => { setMethod(event.target.value); setPlan(null); setSavedExperimentId(""); }}><option value="auto">Automatic</option><option value="qlora">QLoRA</option><option value="lora">LoRA</option><option value="full">Full fine-tuning</option></select></label>
        <p className="quiet-note">The estimate uses detected GPU memory when available. AXIOM never starts a training run from this page.</p>
        {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
        <button type="button" className="button primary-button" onClick={() => void createPlan()} disabled={busy}><Play size={15} /> {busy ? "Building plan…" : "Generate plan"}</button>
      </Surface>
      <Surface title="Plan result" eyebrow={plan ? "ESTIMATE · NOT A BENCHMARK" : "READY WHEN YOU ARE"}>
        {plan ? <>
          <div className="diagnostic-row"><span className={`diagnostic-icon ${plan.fits_hardware ? "success" : "warning"}`}><Cpu size={16} /></span><div><b>{plan.fits_hardware ? "Estimated to fit" : "Hardware fit needs review"}</b><span>{plan.reason}</span></div><StatusPill tone={plan.fits_hardware ? "success" : "warning"}>{plan.fits_hardware ? "ESTIMATE" : "REVIEW"}</StatusPill></div>
          <div className="plan-result-grid">
            <div><small>Precision</small><b>{plan.precision}</b></div>
            <div><small>Estimated VRAM</small><b>{plan.estimated_vram_gb.toFixed(2)} GB</b></div>
            <div><small>Batch size</small><b>{plan.batch_size}</b></div>
            <div><small>Gradient accumulation</small><b>{plan.gradient_accumulation}</b></div>
            <div><small>Sequence length</small><b>{plan.sequence_length.toLocaleString()}</b></div>
            <div><small>Learning rate</small><b>{plan.learning_rate}</b></div>
          </div>
          <CliReference command={`axiom train plan ${parameterBillions} --method ${plan.method}`} />
          <div className="experiment-save-row">
            <button type="button" className="button secondary-button" onClick={() => void saveExperiment()} disabled={savingExperiment}>
              {savingExperiment ? "Saving plan…" : savedExperimentId ? "Save another snapshot" : "Save to experiment history"}
            </button>
            {savedExperimentId && <span className="quiet-note">Saved · {savedExperimentId.slice(0, 12)}</span>}
          </div>
        </> : <EmptyState icon={Hammer} title="No plan generated" description="Choose a model size and method to receive a reproducible hardware-aware starting point." action={<CliReference command="axiom train plan 7 --method qlora" />} />}
      </Surface>
    </div>
  </FeaturePage>;
}

function Evaluation({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [experiments, setExperiments] = useState<WorkspaceExperiment[]>([]);
  const [datasets, setDatasets] = useState<WorkspaceDataset[]>([]);
  const [evaluationRuns, setEvaluationRuns] = useState<WorkspaceEvaluationRun[]>([]);
  const [baselineRuns, setBaselineRuns] = useState<WorkspaceEvaluationRun[]>([]);
  const [evaluationTotal, setEvaluationTotal] = useState(0);
  const [evaluationOffset, setEvaluationOffset] = useState(0);
  const [datasetPath, setDatasetPath] = useState("");
  const [predictionField, setPredictionField] = useState("prediction");
  const [referenceField, setReferenceField] = useState("reference");
  const [splitField, setSplitField] = useState("");
  const [splitValue, setSplitValue] = useState("");
  const [modelLabel, setModelLabel] = useState("");
  const [markBaseline, setMarkBaseline] = useState(false);
  const [baselineRunId, setBaselineRunId] = useState("");
  const [regressionMetric, setRegressionMetric] = useState<"exact_match" | "normalized_exact_match" | "token_f1_micro">("token_f1_micro");
  const [regressionTolerance, setRegressionTolerance] = useState("0.05");
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [deletingRun, setDeletingRun] = useState<WorkspaceEvaluationRun | null>(null);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [tagFilter, setTagFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<WorkspaceExperiment | null>(null);
  const [deleting, setDeleting] = useState<WorkspaceExperiment | null>(null);
  const [editName, setEditName] = useState("");
  const [editNotes, setEditNotes] = useState("");
  const [editTags, setEditTags] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const refresh = async (nextOffset = offset) => {
    setLoading(true);
    setError("");
    try {
      const page = await axiomApi.experimentsPage({
        limit: 20,
        offset: nextOffset,
        q: query.trim() || undefined,
        tag: tagFilter.trim() || undefined,
      });
      setExperiments(page.experiments);
      setTotal(page.total);
      setOffset(nextOffset);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment history unavailable.");
    } finally {
      setLoading(false);
    }
  };

  const refreshEvaluationRuns = async (nextOffset = evaluationOffset) => {
    try {
      const [page, allRuns] = await Promise.all([
        axiomApi.evaluationRunsPage(20, nextOffset),
        axiomApi.evaluationRunsPage(1000, 0),
      ]);
      setEvaluationRuns(page.runs);
      setEvaluationTotal(page.total);
      setEvaluationOffset(nextOffset);
      setBaselineRuns(allRuns.runs.filter((run) => run.baseline));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Evaluation reports unavailable.");
    }
  };

  useEffect(() => {
    axiomApi.datasets().then((items) => {
      setDatasets(items);
      setDatasetPath((current) => current || items[0]?.path || "");
    }).catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : "Dataset inventory unavailable.");
    });
    void refreshEvaluationRuns(0);
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(0), 180);
    return () => window.clearTimeout(timer);
  }, [query, tagFilter]);

  const beginEdit = (experiment: WorkspaceExperiment) => {
    setEditing(experiment);
    setEditName(experiment.name);
    setEditNotes(experiment.notes ?? "");
    setEditTags(experiment.tags.join(", "));
    setError("");
  };

  const saveMetadata = async () => {
    if (!editing) return;
    setSaving(true);
    setError("");
    try {
      await axiomApi.updateExperiment(editing.id, {
        name: editName,
        notes: editNotes || null,
        tags: [...new Set(editTags.split(",").map((tag) => tag.trim()).filter(Boolean))],
      });
      setEditing(null);
      onNotify("Experiment metadata updated. The saved plan snapshot is unchanged.", "success");
      await refresh(offset);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment metadata could not be updated.");
    } finally {
      setSaving(false);
    }
  };

  const removeExperiment = async () => {
    if (!deleting) return;
    setSaving(true);
    setError("");
    try {
      await axiomApi.deleteExperiment(deleting.id);
      setDeleting(null);
      onNotify("Saved plan record deleted. No run artifacts were removed.", "success");
      await refresh(experiments.length === 1 && offset > 0 ? offset - 20 : offset);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment record could not be deleted.");
    } finally {
      setSaving(false);
    }
  };

  const runOfflineEvaluation = async () => {
    if (!datasetPath || !predictionField.trim() || !referenceField.trim()) return;
    setEvaluationBusy(true);
    setError("");
    try {
      const result = await axiomApi.createEvaluationRun({
        path: datasetPath,
        prediction_field: predictionField.trim(),
        reference_field: referenceField.trim(),
        split_field: splitField.trim() || undefined,
        split_value: splitValue.trim() || undefined,
        model_label: modelLabel.trim() || undefined,
        baseline: markBaseline,
        baseline_run_id: baselineRunId || undefined,
        regression_metric: regressionMetric,
        regression_tolerance: Number(regressionTolerance),
      });
      onNotify(
        result.run.comparison?.status === "regression"
          ? `Regression gate failed: ${result.run.comparison.delta.toFixed(3)} ${result.run.comparison.metric}.`
          : "Offline evaluation report saved. No model inference was run.",
        result.run.comparison?.status === "regression" ? "warning" : "success",
      );
      await refreshEvaluationRuns(0);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Offline evaluation could not be completed.");
    } finally {
      setEvaluationBusy(false);
    }
  };

  const removeEvaluationRun = async () => {
    if (!deletingRun) return;
    setSaving(true);
    setError("");
    try {
      await axiomApi.deleteEvaluationRun(deletingRun.id);
      setDeletingRun(null);
      onNotify("Evaluation report deleted from local history.", "success");
      await refreshEvaluationRuns(evaluationRuns.length === 1 && evaluationOffset > 0 ? evaluationOffset - 20 : evaluationOffset);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Evaluation report could not be deleted.");
    } finally {
      setSaving(false);
    }
  };

  const exportEvaluationRun = (run: WorkspaceEvaluationRun) => {
    const blob = new Blob([JSON.stringify(run, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `axiom-evaluation-${run.id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return <FeaturePage page="evaluation" actions={<button type="button" className="button secondary-button" onClick={() => { void refresh(); void refreshEvaluationRuns(); }} disabled={loading}><RefreshCcw size={16} /> {loading ? "Refreshing" : "Refresh history"}</button>}>
    <div className="info-grid">
      <InfoCard icon={BarChart3} label="Saved plans" value={loading ? "—" : String(total)} detail="Persistent local snapshots" tone={total ? "success" : "neutral"} />
      <InfoCard icon={Gauge} label="Offline evaluations" value={String(evaluationTotal)} detail="Scored from supplied predictions" tone={evaluationTotal ? "success" : "neutral"} />
      <InfoCard icon={TriangleAlert} label="Baselines" value={String(baselineRuns.length)} detail="Immutable local references" tone={baselineRuns.length ? "success" : "neutral"} />
    </div>
    <Surface title="Evaluate existing predictions" eyebrow="BOUNDED / LOCAL / NO INFERENCE">
      {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
      <div className="two-column-grid">
        <div>
          <label className="field"><span className="field-label-row"><span>JSONL dataset</span></span><select value={datasetPath} onChange={(event) => setDatasetPath(event.target.value)}><option value="">{datasets.length ? "Select a dataset" : "No JSONL files under data/"}</option>{datasets.map((dataset) => <option value={dataset.path} key={dataset.path}>{dataset.path} · {formatBytes(dataset.size_bytes)}</option>)}</select></label>
          <label className="field"><span className="field-label-row"><span>Prediction field</span></span><input value={predictionField} onChange={(event) => setPredictionField(event.target.value)} maxLength={160} /></label>
          <label className="field"><span className="field-label-row"><span>Reference field</span></span><input value={referenceField} onChange={(event) => setReferenceField(event.target.value)} maxLength={160} /></label>
          <label className="field"><span className="field-label-row"><span>Split field (optional)</span></span><input value={splitField} onChange={(event) => setSplitField(event.target.value)} maxLength={160} placeholder="split" /></label>
          <label className="field"><span className="field-label-row"><span>Selected split value</span></span><input value={splitValue} onChange={(event) => setSplitValue(event.target.value)} maxLength={160} placeholder="test" /></label>
        </div>
        <div>
          <label className="field"><span className="field-label-row"><span>Model label (optional)</span></span><input value={modelLabel} onChange={(event) => setModelLabel(event.target.value)} maxLength={256} placeholder="candidate model" /></label>
          <label className="field"><span className="field-label-row"><span>Compare against baseline</span></span><select value={baselineRunId} onChange={(event) => setBaselineRunId(event.target.value)}><option value="">No baseline comparison</option>{baselineRuns.map((run) => <option value={run.id} key={run.id}>{run.name} · {run.report.metrics.token_f1_micro.toFixed(3)} token F1</option>)}</select></label>
          <label className="field"><span className="field-label-row"><span>Regression metric</span></span><select value={regressionMetric} onChange={(event) => setRegressionMetric(event.target.value as typeof regressionMetric)}><option value="token_f1_micro">Token F1 (micro)</option><option value="exact_match">Exact match</option><option value="normalized_exact_match">Normalized exact match</option></select></label>
          <label className="field"><span className="field-label-row"><span>Allowed score drop</span></span><input type="number" min="0" max="1" step="0.01" value={regressionTolerance} onChange={(event) => setRegressionTolerance(event.target.value)} /></label>
          <label className="checkbox-field"><input type="checkbox" checked={markBaseline} onChange={(event) => setMarkBaseline(event.target.checked)} /><span>Mark this immutable report as a baseline</span></label>
          <p className="quiet-note">Metrics use exact match, whitespace-normalized case-insensitive exact match, and micro token F1. Only aggregate counts and scores are stored; raw text is never returned.</p>
          <button type="button" className="button primary-button" onClick={() => void runOfflineEvaluation()} disabled={evaluationBusy || !datasetPath || !predictionField.trim() || !referenceField.trim() || Boolean(splitField.trim()) !== Boolean(splitValue.trim())}>{evaluationBusy ? "Scoring bounded dataset…" : "Run offline evaluation"}</button>
        </div>
      </div>
    </Surface>
    <Surface title="Evaluation reports" eyebrow="IMMUTABLE / LOCAL HISTORY">
      {evaluationRuns.length ? <div className="module-list">{evaluationRuns.map((run) => <div className="module-row experiment-row" key={run.id}>
        <span className="module-icon"><BarChart3 size={17} /></span>
        <span className="module-copy"><b>{run.name}{run.baseline ? " · baseline" : ""}</b><small>{run.report.path} · {run.report.counts.valid_samples.toLocaleString()} samples · exact {run.report.metrics.exact_match.toFixed(3)} (95% {run.report.confidence_intervals.exact_match.low.toFixed(3)}–{run.report.confidence_intervals.exact_match.high.toFixed(3)}) · normalized {run.report.metrics.normalized_exact_match.toFixed(3)} · token F1 {run.report.metrics.token_f1_micro.toFixed(3)} · {run.status}{run.comparison ? ` · ${run.comparison.status} vs baseline (${run.comparison.delta >= 0 ? "+" : ""}${run.comparison.delta.toFixed(3)})` : ""}</small></span>
        <StatusPill tone={run.comparison?.status === "regression" ? "warning" : run.baseline ? "success" : "neutral"}>{run.comparison?.status === "regression" ? "REGRESSION" : run.baseline ? "BASELINE" : run.status.toUpperCase()}</StatusPill>
        <button type="button" className="icon-button" aria-label={"Export " + run.name + " report as JSON"} onClick={() => exportEvaluationRun(run)}><ArrowDownToLine size={15} /></button>
        <button type="button" className="icon-button danger-icon-button" aria-label={"Delete " + run.name + " evaluation report"} onClick={() => setDeletingRun(run)}><Trash2 size={15} /></button>
      </div>)}</div> : <EmptyState icon={BarChart3} title="No evaluation reports" description="Select a local JSONL file containing precomputed prediction and reference fields. AXIOM does not run model inference here." action={<CliReference command="axiom evaluation run ./data/eval.jsonl --prediction-field prediction --reference-field reference" />} />}
      {evaluationTotal > 20 && <div className="pagination-actions"><button type="button" className="button secondary-button" disabled={evaluationOffset === 0} onClick={() => void refreshEvaluationRuns(Math.max(0, evaluationOffset - 20))}>Previous</button><small>{evaluationOffset + 1}–{Math.min(evaluationOffset + evaluationRuns.length, evaluationTotal)} of {evaluationTotal}</small><button type="button" className="button secondary-button" disabled={evaluationOffset + evaluationRuns.length >= evaluationTotal} onClick={() => void refreshEvaluationRuns(evaluationOffset + 20)}>Next</button></div>}
    </Surface>
    <Surface title="Experiment history" eyebrow="PERSISTENT / LOCAL">
      {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
      <div className="experiment-filters">
        <label className="event-filter"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, model, dataset, notes, or tags" aria-label="Search experiments" /></label>
        <label className="event-filter"><Tag size={15} /><input value={tagFilter} onChange={(event) => setTagFilter(event.target.value)} placeholder="Filter by exact tag" aria-label="Filter experiments by tag" /></label>
      </div>
      {loading ? <div className="loading-state" role="status">Loading saved experiments…</div> : experiments.length ? <div className="module-list">{experiments.map((experiment) => <div className="module-row experiment-row" key={experiment.id}>
        <span className="module-icon"><CircleGauge size={17} /></span>
        <span className="module-copy"><b>{experiment.name}</b><small>{experiment.parameter_billions}B parameters · requested {experiment.requested_method} · estimate {experiment.plan.method}/{experiment.plan.precision} · {experiment.hardware?.gpu_name || "CPU"} · {formatTimestamp(experiment.created_at)}{experiment.tags.length ? " · tags: " + experiment.tags.join(", ") : ""}</small></span>
        <StatusPill tone="neutral">PLANNED</StatusPill>
        <button type="button" className="icon-button" aria-label={"Edit " + experiment.name + " metadata"} onClick={() => beginEdit(experiment)}><Pencil size={15} /></button>
        <button type="button" className="icon-button danger-icon-button" aria-label={"Delete " + experiment.name + " plan"} onClick={() => setDeleting(experiment)}><Trash2 size={15} /></button>
      </div>)}</div> : <EmptyState icon={CircleGauge} title="No saved experiments" description="Build a hardware-aware plan in Training, then save it here. Saved records describe plans only; they do not represent completed runs." action={<CliReference command="axiom experiment list" />} />}
      {total > 20 && <div className="pagination-actions"><button type="button" className="button secondary-button" disabled={offset === 0 || loading} onClick={() => void refresh(Math.max(0, offset - 20))}>Previous</button><small>{offset + 1}–{Math.min(offset + experiments.length, total)} of {total}</small><button type="button" className="button secondary-button" disabled={offset + experiments.length >= total || loading} onClick={() => void refresh(offset + 20)}>Next</button></div>}
    </Surface>
    <Callout tone="info" title="Precomputed outputs only">Evaluation scores compare existing reference and prediction text. AXIOM does not execute a model, and these text metrics are not a substitute for task-specific human review.</Callout>
    {editing && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !saving) setEditing(null); }}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="edit-experiment-title"><div className="dialog-head"><div><p className="eyebrow">METADATA ONLY</p><h2 id="edit-experiment-title">Edit {editing.name}</h2></div><button type="button" className="icon-button" aria-label="Close edit dialog" onClick={() => setEditing(null)} disabled={saving}><X size={16} /></button></div><p className="quiet-note">Editing does not change the saved hardware-aware plan snapshot.</p><label className="field"><span className="field-label-row"><span>Name</span></span><input value={editName} onChange={(event) => setEditName(event.target.value)} maxLength={120} /></label><label className="field"><span className="field-label-row"><span>Notes</span></span><textarea value={editNotes} onChange={(event) => setEditNotes(event.target.value)} maxLength={1000} rows={3} /></label><label className="field"><span className="field-label-row"><span>Tags, comma separated</span></span><input value={editTags} onChange={(event) => setEditTags(event.target.value)} maxLength={1400} placeholder="baseline, review" /></label><div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => setEditing(null)} disabled={saving}>Cancel</button><button type="button" className="button primary-button" onClick={() => void saveMetadata()} disabled={saving || !editName.trim()}>{saving ? "Saving…" : "Save metadata"}</button></div></section></div>}
    {deleting && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !saving) setDeleting(null); }}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="delete-experiment-title"><div className="dialog-head"><div><p className="eyebrow">CONFIRM DELETION</p><h2 id="delete-experiment-title">Delete {deleting.name}?</h2></div><button type="button" className="icon-button" aria-label="Close delete confirmation" onClick={() => setDeleting(null)} disabled={saving}><X size={16} /></button></div><p className="quiet-note">This removes the saved plan record only. No model, dataset, or run artifacts are deleted.</p><div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => setDeleting(null)} disabled={saving}>Cancel</button><button type="button" className="button danger-button" onClick={() => void removeExperiment()} disabled={saving}>{saving ? "Deleting…" : "Delete plan record"}</button></div></section></div>}
    {deletingRun && <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !saving) setDeletingRun(null); }}><section className="dialog" role="dialog" aria-modal="true" aria-labelledby="delete-evaluation-title"><div className="dialog-head"><div><p className="eyebrow">CONFIRM DELETION</p><h2 id="delete-evaluation-title">Delete {deletingRun.name}?</h2></div><button type="button" className="icon-button" aria-label="Close delete confirmation" onClick={() => setDeletingRun(null)} disabled={saving}><X size={16} /></button></div><p className="quiet-note">This removes only the saved aggregate report. The source dataset is unchanged.</p><div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => setDeletingRun(null)} disabled={saving}>Cancel</button><button type="button" className="button danger-button" onClick={() => void removeEvaluationRun()} disabled={saving}>{saving ? "Deleting…" : "Delete report"}</button></div></section></div>}
  </FeaturePage>;
}

const workflowNodeLabels: Record<WorkflowNodeType, string> = {
  "dataset.profile": "Profile dataset",
  "model.inspect": "Inspect model",
  "training.plan": "Create training plan",
  "evaluation.run": "Evaluation step",
  "report.export": "Export report",
};

function Workflows({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [workflows, setWorkflows] = useState<WorkspaceWorkflow[]>([]);
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [templateId, setTemplateId] = useState("");
  const [name, setName] = useState("New local workflow");
  const [nodes, setNodes] = useState<WorkflowNode[]>([]);
  const [edges, setEdges] = useState<Array<{ source: string; target: string }>>([]);
  const [nodeType, setNodeType] = useState<WorkflowNodeType>("dataset.profile");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [validation, setValidation] = useState("");
  const [error, setError] = useState("");
  const [dryRun, setDryRun] = useState<WorkflowDryRun | null>(null);
  const [dryRunBusy, setDryRunBusy] = useState(false);
  const [dryRunParameters, setDryRunParameters] = useState("7");
  const [dryRunMethod, setDryRunMethod] = useState("auto");
  const [dryRunSeed, setDryRunSeed] = useState("42");
  const [autosaveEnabled, setAutosaveEnabled] = useState(true);
  const selectedWorkflow = workflows.find((workflow) => workflow.id === workflowId);
  const workflowDirty = Boolean(selectedWorkflow && (
    name !== selectedWorkflow.name
    || JSON.stringify(nodes) !== JSON.stringify(selectedWorkflow.nodes)
    || JSON.stringify(edges) !== JSON.stringify(selectedWorkflow.edges)
  ));

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const [savedWorkflows, availableTemplates] = await Promise.all([
        axiomApi.workflows(),
        axiomApi.workflowTemplates(),
      ]);
      setWorkflows(savedWorkflows);
      setTemplates(availableTemplates);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Workflow list unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!autosaveEnabled || !workflowId || !workflowDirty || saving) return;
    const timeout = window.setTimeout(() => {
      setSaving(true);
      setError("");
      void axiomApi.updateWorkflow(workflowId, { name, nodes, edges })
        .then(({ workflow }) => {
          setWorkflows((current) => current.map((item) => item.id === workflow.id ? workflow : item));
          setValidation(`Autosaved · revision ${workflow.revision} · design only`);
          setDryRun(null);
        })
        .catch((cause: unknown) => {
          setError(cause instanceof Error ? cause.message : "Workflow autosave failed.");
          setAutosaveEnabled(false);
        })
        .finally(() => setSaving(false));
    }, 900);
    return () => window.clearTimeout(timeout);
  }, [autosaveEnabled, workflowId, workflowDirty, saving, name, nodes, edges]);

  const startNew = () => {
    setWorkflowId(null);
    setName("New local workflow");
    setNodes([]);
    setEdges([]);
    setValidation("");
    setError("");
    setDryRun(null);
  };

  const selectWorkflow = (id: string) => {
    const selected = workflows.find((item) => item.id === id);
    if (!selected) return;
    setWorkflowId(selected.id);
    setName(selected.name);
    setNodes(selected.nodes);
    setEdges(selected.edges);
    setValidation("");
    setError("");
    setDryRun(null);
  };

  const loadTemplate = () => {
    const template = templates.find((item) => item.id === templateId);
    if (!template) return;
    setWorkflowId(null);
    setName(template.name);
    setNodes(template.nodes.map((node) => ({ ...node })));
    setEdges(template.edges.map((edge) => ({ ...edge })));
    setValidation("");
    setError("");
    setDryRun(null);
  };

  const duplicateWorkflow = async () => {
    if (!workflowId) return;
    setSaving(true);
    setError("");
    try {
      const result = await axiomApi.duplicateWorkflow(workflowId);
      setWorkflowId(result.workflow.id);
      setName(result.workflow.name);
      setNodes(result.workflow.nodes);
      setEdges(result.workflow.edges);
      setValidation("Copied as a new revision-1 design");
      setDryRun(null);
      onNotify("Workflow duplicated locally.", "success");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Workflow could not be duplicated.");
    } finally {
      setSaving(false);
    }
  };

  const addNode = () => {
    if (nodes.length >= 64) return;
    const id = `step-${crypto.randomUUID()}`;
    const previous = nodes.at(-1);
    setNodes((current) => [...current, { id, type: nodeType }]);
    if (previous) setEdges((current) => [...current, { source: previous.id, target: id }]);
    setValidation("");
    setDryRun(null);
  };

  const removeNode = (id: string) => {
    const remaining = nodes.filter((node) => node.id !== id);
    setNodes(remaining);
    setEdges(remaining.slice(1).map((node, index) => ({ source: remaining[index].id, target: node.id })));
    setValidation("");
    setDryRun(null);
  };

  const runDryRun = async () => {
    if (!workflowId || workflowDirty) return;
    const parameters = Number(dryRunParameters);
    if (!Number.isFinite(parameters) || parameters <= 0 || parameters > 1000) {
      onNotify("Model size must be between 0 and 1000 billion parameters.", "warning");
      return;
    }
    const seed = Number(dryRunSeed);
    if (!Number.isInteger(seed) || seed < 0 || seed > 4_294_967_295) {
      onNotify("Seed must be an integer between 0 and 4,294,967,295.", "warning");
      return;
    }
    setDryRunBusy(true);
    setError("");
    try {
      const result = await axiomApi.dryRunWorkflow(workflowId, {
        parameter_billions: parameters,
        method: dryRunMethod,
        seed,
      });
      setDryRun(result.dry_run);
      onNotify("Workflow dry run completed. No job was created or executed.", "success");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Workflow dry run failed.");
    } finally {
      setDryRunBusy(false);
    }
  };

  const validateGraph = async () => {
    setError("");
    setValidation("");
    try {
      const result = await axiomApi.validateWorkflow({ name, nodes, edges });
      setValidation(`${result.node_count} steps · ${result.connection_count} connections · no cycles`);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Workflow validation failed.");
      return false;
    }
  };

  const saveWorkflow = async () => {
    setSaving(true);
    const valid = await validateGraph();
    if (!valid) {
      setSaving(false);
      return;
    }
    try {
      const payload = { name, nodes, edges };
      const result = workflowId
        ? await axiomApi.updateWorkflow(workflowId, payload)
        : await axiomApi.createWorkflow(payload);
      setWorkflowId(result.workflow.id);
      setDryRun(null);
      setValidation("Saved · design only · execution is disabled");
      onNotify("Workflow saved. It is validated but cannot execute yet.", "success");
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Workflow could not be saved.");
    } finally {
      setSaving(false);
    }
  };

  const exportReproducibilityManifest = () => {
    if (!dryRun) return;
    const blob = new Blob([JSON.stringify(dryRun.reproducibility_manifest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `workflow-${dryRun.workflow_id}-reproducibility.json`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  };

  return <FeaturePage page="workflows" actions={<><button type="button" className="button secondary-button" onClick={() => void duplicateWorkflow()} disabled={!workflowId || saving}><Copy size={15} /> Duplicate selected</button><button type="button" className="button secondary-button" onClick={startNew}><Plus size={16} /> New workflow</button></>}>
    <div className="info-grid">
      <InfoCard icon={GitBranch} label="Saved graphs" value={loading ? "—" : String(workflows.length)} detail="Persistent local workflows" tone={workflows.length ? "success" : "neutral"} />
      <InfoCard icon={CircleCheck} label="Validation" value={validation ? "Passed" : "On demand"} detail={validation || "Backend checks graph structure"} tone={validation ? "success" : "neutral"} />
      <InfoCard icon={TriangleAlert} label="Execution" value="Disabled" detail="No training runner is attached" tone="warning" />
    </div>
    <div className="two-column-grid workflow-layout">
      <Surface title="Workflow graph" eyebrow="DESIGN / VALIDATE / SAVE" action={<StatusPill tone="neutral">NOT EXECUTABLE</StatusPill>}>
        <div className="workflow-template-row">
          <label className="field"><span className="field-label-row"><span>Start from template</span></span><select value={templateId} onChange={(event) => setTemplateId(event.target.value)}><option value="">Choose a template</option>{templates.map((template) => <option key={template.id} value={template.id}>{template.name}</option>)}</select></label>
          <button type="button" className="button secondary-button" onClick={loadTemplate} disabled={!templateId}><FileCode2 size={15} /> Load template</button>
        </div>
        {templateId && <p className="quiet-note">{templates.find((template) => template.id === templateId)?.description}</p>}
        <label className="field"><span className="field-label-row"><span>Workflow name</span></span><input value={name} maxLength={120} onChange={(event) => { setName(event.target.value); setValidation(""); setDryRun(null); }} /></label>
        <div className="workflow-builder-controls">
          <label className="field"><span className="field-label-row"><span>Add step</span></span><select value={nodeType} onChange={(event) => setNodeType(event.target.value as WorkflowNodeType)}>{Object.entries(workflowNodeLabels).map(([type, label]) => <option key={type} value={type}>{label}</option>)}</select></label>
          <button type="button" className="button secondary-button" onClick={addNode} disabled={nodes.length >= 64}><Plus size={15} /> {nodes.length >= 64 ? "Step limit reached" : "Add step"}</button>
        </div>
        <div className="workflow-dry-run-controls">
          <label className="field"><span className="field-label-row"><span>Estimate model size (billions)</span></span><input inputMode="decimal" value={dryRunParameters} onChange={(event) => { setDryRunParameters(event.target.value); setDryRun(null); }} maxLength={8} /></label>
          <label className="field"><span className="field-label-row"><span>Planning method</span></span><select value={dryRunMethod} onChange={(event) => { setDryRunMethod(event.target.value); setDryRun(null); }}><option value="auto">Auto</option><option value="qlora">QLoRA</option><option value="lora">LoRA</option><option value="full">Full fine-tune</option></select></label>
          <label className="field"><span className="field-label-row"><span>Planned seed</span></span><input inputMode="numeric" value={dryRunSeed} onChange={(event) => { setDryRunSeed(event.target.value); setDryRun(null); }} maxLength={10} /></label>
        </div>
        <label className="workflow-autosave-toggle"><input type="checkbox" checked={autosaveEnabled} onChange={(event) => setAutosaveEnabled(event.target.checked)} /><span><b>Autosave selected workflow</b><small>Existing saved designs update locally 900 ms after an edit; new workflows still require Save workflow.</small></span></label>
        {nodes.length ? <div className="workflow-canvas" aria-label="Workflow steps">{nodes.map((node, index) => <Fragment key={node.id}>
          {index > 0 && <div className="workflow-connector" aria-hidden="true"><ArrowRight size={16} /></div>}
          <div className="workflow-node"><span className="workflow-node-number">{String(index + 1).padStart(2, "0")}</span><b>{workflowNodeLabels[node.type]}</b><button type="button" className="workflow-node-remove" onClick={() => removeNode(node.id)} aria-label={`Remove ${workflowNodeLabels[node.type]} step`}><X size={13} /></button><small>{node.type}</small></div>
        </Fragment>)}</div> : <EmptyState icon={GitBranch} title="Start with a workflow step" description="Add typed steps and the builder will connect them in order. The backend checks IDs, references, duplicates, size, and cycles." />}
        {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
        {validation && <InlineAlert tone="success">{validation}</InlineAlert>}
        <div className="integration-form-actions">
          <button type="button" className="button secondary-button" onClick={() => void runDryRun()} disabled={!workflowId || workflowDirty || saving || dryRunBusy}>{dryRunBusy ? "Estimating…" : workflowDirty ? "Save to dry-run changes" : "Dry-run saved graph"}</button>
          <button type="button" className="button secondary-button" onClick={() => void validateGraph()} disabled={saving}>Validate graph</button>
          <button type="button" className="button primary-button" onClick={() => void saveWorkflow()} disabled={saving}>{saving ? "Saving…" : workflowId ? "Save changes" : "Save workflow"}</button>
        </div>
      </Surface>
      <Surface title="Saved workflows" eyebrow="LOCAL WORKSPACE">
        {loading ? <div className="loading-state" role="status">Loading workflows…</div> : error && !workflows.length ? <InlineAlert tone="warning" role="alert">{error}</InlineAlert> : workflows.length ? <div className="module-list">{workflows.map((workflow) => <button type="button" className={`module-row workflow-list-item ${workflow.id === workflowId ? "selected" : ""}`} key={workflow.id} onClick={() => selectWorkflow(workflow.id)}><span className="module-icon"><GitBranch size={17} /></span><span className="module-copy"><b>{workflow.name}</b><small>{workflow.nodes.length} steps · revision {workflow.revision} · updated {formatTimestamp(workflow.updated_at)}</small></span><StatusPill tone="neutral">DESIGN</StatusPill></button>)}</div> : <EmptyState icon={GitBranch} title="No saved workflows" description="Create and validate a workflow graph; saved graphs are local JSON metadata." />}
        {selectedWorkflow && <details className="workflow-history"><summary>Revision history · {selectedWorkflow.versions.length} saved versions</summary><div className="workflow-version-list"><div><b>Current · r{selectedWorkflow.revision}</b><small>{selectedWorkflow.name} · {formatTimestamp(selectedWorkflow.updated_at)}</small></div>{[...selectedWorkflow.versions].reverse().map((version) => <div key={version.revision}><b>Revision {version.revision}</b><small>{version.name} · {formatTimestamp(version.updated_at)} · {version.nodes.length} steps</small></div>)}</div></details>}
      </Surface>
    </div>
    {dryRun && <Surface title="Workflow dry-run summary" eyebrow={`PLANNING ONLY · REVISION ${dryRun.workflow_revision}`} action={<div className="dataset-page-actions"><button type="button" className="button secondary-button" onClick={exportReproducibilityManifest}><ArrowDownToLine size={15} /> Export manifest</button><StatusPill tone="warning">NOT EXECUTED</StatusPill></div>}>
      <p className="quiet-note">No training, evaluation, or report job was created. This summary uses current hardware detection and the requested planning settings.</p>
      <div className="info-grid">
        <InfoCard icon={Gauge} label="Estimated VRAM" value={dryRun.resource_summary.estimated_vram_gb === null ? "Not needed" : `${dryRun.resource_summary.estimated_vram_gb.toFixed(1)} GB`} detail={dryRun.resource_summary.method || "No training-plan node"} tone="neutral" />
        <InfoCard icon={CircleAlert} label="Hardware fit" value={dryRun.resource_summary.fits_hardware === null ? "N/A" : dryRun.resource_summary.fits_hardware ? "Estimated fit" : "May not fit"} detail={`${dryRun.resource_summary.available_hardware.ram_gb.toFixed(1)} GB system RAM · ${dryRun.resource_summary.available_hardware.vram_gb === null ? "No detected GPU VRAM" : `${dryRun.resource_summary.available_hardware.vram_gb.toFixed(1)} GB GPU VRAM`}`} tone={dryRun.resource_summary.fits_hardware === false ? "warning" : "success"} />
        <InfoCard icon={GitBranch} label="Graph steps" value={String(dryRun.node_count)} detail={`${dryRun.unresolved_inputs.length} nodes need concrete inputs`} tone="neutral" />
      </div>
      {dryRun.resource_summary.estimated_training_plan && <div className="dataset-quality-check passed"><b>Training-plan estimate · {dryRun.resource_summary.parameters_billions}B · {dryRun.resource_summary.method}</b><small>{dryRun.resource_summary.estimated_training_plan.reason}</small></div>}
      <div className="dataset-quality-check passed"><b>Reproducibility manifest · seed {dryRun.reproducibility_manifest.seed} (not applied)</b><small>Graph SHA-256: {dryRun.reproducibility_manifest.graph_sha256} · AXIOM {dryRun.reproducibility_manifest.software.axiom} · Python {dryRun.reproducibility_manifest.software.python}</small><small>Dependency versions and detected hardware are included in the downloaded JSON. Seed remains unapplied because execution is disabled.</small></div>
      {dryRun.unresolved_inputs.map((input) => <p className="quiet-note" key={input.node_id}>{input.node_id} · {workflowNodeLabels[input.type]}: {input.reason}</p>)}
      <div className="module-list">{dryRun.steps.map((step) => <div className="module-row" key={step.node_id}><span className="module-icon"><CircleCheck size={16} /></span><span className="module-copy"><b>{step.node_id} · {workflowNodeLabels[step.type]}</b><small>Design step only · state: {step.state}</small></span><StatusPill tone="neutral">NOT RUN</StatusPill></div>)}</div>
      {dryRun.warnings.map((warning) => <p className="quiet-note" key={warning}>{warning}</p>)}
    </Surface>}
  </FeaturePage>;
}

function Runtime() {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      setSnapshot(await axiomApi.runtime());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Runtime status unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);
  const hardware = snapshot?.hardware;
  const statusTone: Tone = snapshot?.status === "online" ? "success" : "warning";

  return <FeaturePage page="runtime" actions={<button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}><RefreshCcw size={16} /> {loading ? "Checking" : "Refresh status"}</button>}>
    <div className="runtime-banner"><div className="runtime-state-icon"><RadioTower size={20} /></div><div><StatusPill tone={statusTone}>{loading ? "CHECKING" : snapshot?.status === "online" ? "ONLINE" : "OFFLINE"}</StatusPill><h2>{snapshot?.status === "online" ? "Runtime connected" : "Inference runtime not connected"}</h2><p>{error || "Hardware detection is live. AXIOM does not currently expose a local inference-serving endpoint."}</p></div><code>axiom mcp serve</code></div>
    <div className="info-grid"><InfoCard icon={Cpu} label="Processor" value={hardware?.architecture || "—"} detail={hardware ? `${hardware.cpu_cores} CPU cores` : loading ? "Checking" : "Unavailable"} tone={hardware ? "success" : "neutral"} /><InfoCard icon={HardDrive} label="System memory" value={hardware ? `${hardware.ram_gb.toFixed(1)} GB` : "—"} detail="Detected locally" tone={hardware ? "success" : "neutral"} /><InfoCard icon={Zap} label="GPU memory" value={hardware?.vram_gb === null || hardware?.vram_gb === undefined ? "Not detected" : `${hardware.vram_gb.toFixed(1)} GB`} detail={hardware?.gpu_name || "No compatible GPU reported"} tone={hardware?.cuda_available ? "success" : "warning"} /></div>
    {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
    <Surface title="Serving status" eyebrow="LOCAL INFERENCE"><EmptyState icon={RadioTower} title="No inference process reported" description="The runtime endpoint currently reports system hardware only. Request volume, latency, and model selection need a serving connector." /></Surface>
    <RuntimeConnectorPanel />
  </FeaturePage>;
}

function MCP({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [policy, setPolicy] = useState<McpToolPolicy | null>(null);
  const [selectedProfile, setSelectedProfile] = useState("default");
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [newProfileName, setNewProfileName] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const result = await axiomApi.mcpToolPolicy();
      setPolicy(result);
      setSelectedProfile(result.active_profile);
      const active = result.profiles.find((profile) => profile.name === result.active_profile);
      setAllowedTools(active?.allowed_tools || []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "MCP permissions unavailable.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  const selectProfile = (name: string) => {
    setSelectedProfile(name);
    setAllowedTools(policy?.profiles.find((profile) => profile.name === name)?.allowed_tools || []);
  };

  const savePermissions = async () => {
    setSaving(true);
    setError("");
    try {
      const result = await axiomApi.saveMcpToolProfile(selectedProfile, allowedTools);
      setPolicy((current) => current ? {
        ...current,
        profiles: current.profiles.map((profile) => profile.name === selectedProfile ? result.profile : profile),
      } : current);
      setAllowedTools(result.profile.allowed_tools);
      onNotify(`Tool permissions saved for ${selectedProfile}.`, "success");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "MCP permissions could not be saved.");
    } finally {
      setSaving(false);
    }
  };

  const activateProfile = async () => {
    setSaving(true);
    setError("");
    try {
      const saved = await axiomApi.saveMcpToolProfile(selectedProfile, allowedTools);
      await axiomApi.setActiveMcpProfile(selectedProfile);
      setPolicy((current) => current ? {
        ...current,
        active_profile: selectedProfile,
        profiles: current.profiles.map((profile) => profile.name === selectedProfile ? saved.profile : profile),
      } : current);
      setAllowedTools(saved.profile.allowed_tools);
      onNotify(`MCP server now enforces ${selectedProfile}.`, "success");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Active MCP profile could not be changed.");
    } finally {
      setSaving(false);
    }
  };

  const createProfile = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      const profileName = newProfileName.trim();
      if (policy?.profiles.some((profile) => profile.name === profileName)) {
        setError("A profile with that name already exists.");
        return;
      }
      const result = await axiomApi.createMcpToolProfile(profileName);
      setNewProfileName("");
      setPolicy((current) => current ? { ...current, profiles: [...current.profiles, result.profile] } : current);
      setSelectedProfile(profileName);
      setAllowedTools([]);
      onNotify(`Created default-deny profile ${profileName}.`, "success");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Profile could not be created.");
    } finally {
      setSaving(false);
    }
  };

  const toggleTool = (name: string, checked: boolean) => {
    setAllowedTools((current) => checked
      ? [...new Set([...current, name])]
      : current.filter((tool) => tool !== name));
  };

  const activeProfile = policy?.active_profile || "—";
  const profileNameExists = policy?.profiles.some((profile) => profile.name === newProfileName.trim()) || false;
  return <FeaturePage page="mcp" actions={<button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}><RefreshCcw size={16} /> {loading ? "Loading" : "Refresh policy"}</button>}>
    <div className="info-grid">
      <InfoCard icon={Network} label="MCP server" value="CLI" detail="Policy enforced per tool call" tone="neutral" />
      <InfoCard icon={ShieldCheck} label="Active profile" value={activeProfile} detail={`${policy?.profiles.find((profile) => profile.name === activeProfile)?.allowed_tools.length || 0} tools allowed`} tone={policy ? "success" : "neutral"} />
      <InfoCard icon={LockKeyhole} label="Default access" value="Deny" detail="No tools granted until explicitly enabled" tone="warning" />
    </div>
    <div className="two-column-grid mcp-layout">
      <Surface title="Tool permissions" eyebrow="BACKEND-ENFORCED">
        {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
        {loading || !policy ? <div className="loading-state" role="status">Loading local MCP policy…</div> : <>
          <label className="field"><span className="field-label-row"><span>Edit profile</span><StatusPill tone={selectedProfile === activeProfile ? "success" : "neutral"}>{selectedProfile === activeProfile ? "ACTIVE" : "INACTIVE"}</StatusPill></span><select value={selectedProfile} onChange={(event) => selectProfile(event.target.value)}>{policy.profiles.map((profile) => <option key={profile.name} value={profile.name}>{profile.name}</option>)}</select></label>
          <div className="permission-list">{policy.tools.map((tool) => <label className="permission-row" key={tool.name}>
            <span><b>{tool.name}</b><small>{tool.description}{tool.name === "axiom_dataset_clean" ? " · writes a cleaned copy" : tool.name === "axiom_supercompress" ? " · may contact an external service" : ""}</small></span>
            <input type="checkbox" checked={allowedTools.includes(tool.name)} onChange={(event) => toggleTool(tool.name, event.target.checked)} aria-label={`Allow ${tool.name}`} />
          </label>)}</div>
          <div className="integration-form-actions"><button type="button" className="button secondary-button" onClick={() => void activateProfile()} disabled={saving || selectedProfile === activeProfile}>Use this profile</button><button type="button" className="button primary-button" onClick={() => void savePermissions()} disabled={saving}>{saving ? "Saving…" : "Save permissions"}</button></div>
        </>}
      </Surface>
      <Surface title="Profiles" eyebrow="LOCAL POLICY">
        <form className="profile-create-form" onSubmit={(event) => void createProfile(event)}>
          <label className="field"><span className="field-label-row"><span>New profile name</span></span><input value={newProfileName} onChange={(event) => setNewProfileName(event.target.value)} minLength={1} maxLength={48} pattern="[A-Za-z0-9_-]+" placeholder="research-agent" required /></label>
          <button type="submit" className="button secondary-button" disabled={saving || !newProfileName.trim() || profileNameExists}><Plus size={15} /> Create denied-by-default</button>
        </form>
        <div className="surface-divider" />
        <Callout tone="warning" title="Enforced by the MCP backend">Changing a checkbox only changes the selected profile after saving. The MCP tool handler reads the active profile for every call; the GUI does not execute tools.</Callout>
        <p className="quiet-note">Use `axiom mcp permissions` to inspect grants, `axiom mcp allow &lt;tool&gt;` to grant a tool, or `axiom mcp profile use &lt;name&gt;` to select a profile.</p>
      </Surface>
    </div>
  </FeaturePage>;
}

function Diagnostics({ onNotify }: { onNotify: (message: string, tone?: Tone) => void }) {
  const [scanState, setScanState] = useState<"idle" | "scanning" | "failed">("scanning");
  const [health, setHealth] = useState<{ online: boolean; version?: string } | null>(null);
  const [hardware, setHardware] = useState<HardwareStatus | null>(null);
  const [modelCount, setModelCount] = useState<number | null>(null);
  const [datasetCount, setDatasetCount] = useState<number | null>(null);
  const [workspaceDiagnostics, setWorkspaceDiagnostics] = useState<WorkspaceDiagnostics | null>(null);
  const [bundleBusy, setBundleBusy] = useState(false);


  const downloadBundle = async () => {
    setBundleBusy(true);
    try {
      const blob = await axiomApi.downloadSupportBundle();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "axiom-support-bundle.zip";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
      onNotify("Privacy-safe support bundle downloaded. Review it before sharing.", "success");
    } catch (cause) {
      onNotify(cause instanceof Error ? cause.message : "Could not download the support bundle.", "warning");
    } finally {
      setBundleBusy(false);
    }
  };

  const runScan = async () => {
    setScanState("scanning");
    const [healthResult, hardwareResult, modelResult, datasetResult, workspaceDiagnosticsResult] = await Promise.allSettled([
      axiomApi.health(),
      axiomApi.optimizationHardware(),
      axiomApi.models(),
      axiomApi.datasets(),
      axiomApi.workspaceDiagnostics(),
    ]);
    const failures: string[] = [];
    if (healthResult.status === "fulfilled") setHealth({ online: healthResult.value.status === "ok", version: healthResult.value.version });
    else { setHealth({ online: false }); failures.push("API health"); }
    if (hardwareResult.status === "fulfilled") setHardware(hardwareResult.value);
    else { setHardware(null); failures.push("hardware"); }
    if (modelResult.status === "fulfilled") setModelCount(modelResult.value.length);
    else { setModelCount(null); failures.push("model registry"); }
    if (datasetResult.status === "fulfilled") setDatasetCount(datasetResult.value.length);
    else { setDatasetCount(null); failures.push("dataset inventory"); }
    if (workspaceDiagnosticsResult.status === "fulfilled") setWorkspaceDiagnostics(workspaceDiagnosticsResult.value);
    else { setWorkspaceDiagnostics(null); failures.push("workspace diagnostics"); }
    setScanState(failures.length ? "failed" : "idle");
    if (failures.length) onNotify(`Some diagnostics could not complete: ${failures.join(", ")}.`, "warning");
  };

  useEffect(() => { void runScan(); }, []);
  const connected = health?.online === true;
  const headline = scanState === "scanning" ? "Checking local services…" : connected ? "Workspace checks complete" : "Backend unavailable";
  const systemDescription = hardware ? `${hardware.cpu_name} · ${hardware.ram_gb ?? "Unknown"} GB RAM · ${hardware.gpu_name || "No GPU reported"}` : "Hardware details unavailable";

  return <FeaturePage page="diagnostics" actions={<>
    <button type="button" className="button secondary-button" onClick={() => void downloadBundle()} disabled={bundleBusy}><ArrowDownToLine size={16} /> {bundleBusy ? "Preparing bundle…" : "Download support bundle"}</button>
    <button type="button" className="button primary-button" onClick={() => void runScan()} disabled={scanState === "scanning"}><Wrench size={16} /> {scanState === "scanning" ? "Scanning…" : "Run diagnostics"}</button>
  </>}>
    <div className="diagnostic-hero"><div><p className="eyebrow">WORKSPACE READINESS</p><strong>{headline}</strong><p>{systemDescription}{health?.version ? ` · AXIOM ${health.version}` : ""}</p></div><div className={`diagnostic-ring ${scanState === "failed" ? "blocked" : scanState}`}><span>{scanState === "scanning" ? "…" : connected ? "OK" : "N/A"}</span></div></div>
    <div className="diag-list">
      <DiagnosticRow icon={connected ? CircleCheck : TriangleAlert} title="AXIOM API" detail={connected ? `Health endpoint responded${health?.version ? ` · version ${health.version}` : ""}.` : scanState === "scanning" ? "Waiting for the local API response." : "Health endpoint did not respond."} state={connected ? "PASS" : scanState === "scanning" ? "CHECKING" : "FAILED"} tone={connected ? "success" : scanState === "scanning" ? "neutral" : "warning"} />
      <DiagnosticRow icon={hardware ? CircleCheck : TriangleAlert} title="Hardware detection" detail={hardware ? `${hardware.cpu_name} · ${hardware.ram_gb ?? "Unknown"} GB RAM · ${hardware.gpu_name}` : "No hardware snapshot is available."} state={hardware ? "PASS" : scanState === "scanning" ? "CHECKING" : "FAILED"} tone={hardware ? "success" : scanState === "scanning" ? "neutral" : "warning"} />
      <DiagnosticRow icon={modelCount !== null ? CircleCheck : TriangleAlert} title="Model registry" detail={modelCount === null ? "Registry endpoint unavailable." : `${modelCount} model${modelCount === 1 ? "" : "s"} registered.`} state={modelCount === null ? "FAILED" : "PASS"} tone={modelCount === null ? "warning" : "success"} />
      <DiagnosticRow icon={datasetCount !== null ? CircleCheck : TriangleAlert} title="Dataset inventory" detail={datasetCount === null ? "Dataset endpoint unavailable." : `${datasetCount} JSONL file${datasetCount === 1 ? "" : "s"} found in data/.`} state={datasetCount === null ? "FAILED" : "PASS"} tone={datasetCount === null ? "warning" : "success"} />
    </div>
    {workspaceDiagnostics && <Surface title="Workspace checks" eyebrow="READ-ONLY BACKEND DIAGNOSTICS" action={<StatusPill tone={workspaceDiagnostics.status === "pass" ? "success" : workspaceDiagnostics.status === "fail" ? "danger" : "warning"}>{workspaceDiagnostics.status.toUpperCase()}</StatusPill>}>
      <p className="quiet-note">{workspaceDiagnostics.summary.passed} passed · {workspaceDiagnostics.summary.warnings} warning(s) · {workspaceDiagnostics.summary.failed} failed. Checks are read-only; no repair actions are run automatically.</p>
      <div className="diag-list">{workspaceDiagnostics.checks.map((check) => {
        const tone: Tone = check.status === "pass" ? "success" : check.status === "fail" ? "danger" : check.status === "warn" ? "warning" : "neutral";
        const Icon = check.status === "pass" ? CircleCheck : check.status === "fail" ? CircleAlert : TriangleAlert;
        return <DiagnosticRow key={check.id} icon={Icon} title={check.title} detail={`${check.detail}${check.remediation ? ` Next: ${check.remediation}` : ""}`} state={check.status.toUpperCase()} tone={tone} />;
      })}</div>
    </Surface>}
    {!connected && scanState !== "scanning" && <Callout tone="warning" title="Start the local API">Open the AXIOM UI launcher or start the FastAPI service, then scan again.</Callout>}
  </FeaturePage>;
}

function ModulesPage({ setPage }: { setPage: (page: Page) => void }) {
  const [query, setQuery] = useState("");
  const [copiedCommand, setCopiedCommand] = useState("");
  type Entry = {
    id: string;
    title: string;
    description: string;
    scope: string;
    availability: string;
    tone: Tone;
    icon: LucideIcon;
    page?: Page;
    command?: string;
    href?: string;
  };
  const entries: Entry[] = [
    { id: "dashboard", title: "Control center", description: "Workspace overview, recent events, and entry points to major product surfaces.", scope: "Workspace", availability: "GUI", tone: "success", icon: LayoutDashboard, page: "dashboard" },
    { id: "optimizer", title: "Agent Optimizer", description: "Questionnaire-driven workload sizing, hardware-aware recommendations, and optimization plans.", scope: "Optimization", availability: "GUI", tone: "success", icon: Zap, page: "optimizer" },
    { id: "quantizer", title: "Quantization Lab", description: "Compare quantization choices and review estimated memory fit before runtime work.", scope: "Optimization", availability: "GUI", tone: "success", icon: SlidersHorizontal, page: "quantizer" },
    { id: "models", title: "Model registry and inspection", description: "Discover local folders, inspect metadata, register entries, manage tags/favorites, and import/export registry data.", scope: "Model engineering", availability: "GUI + CLI", tone: "success", icon: BrainCircuit, page: "models", command: "axiom model --help" },
    { id: "datasets", title: "Dataset engineering and quality", description: "Inspect and validate JSONL, profile schema and duplicates, compare datasets, preview cleaning, and review manifests/privacy checks.", scope: "Data engineering", availability: "GUI + CLI", tone: "success", icon: Database, page: "datasets", command: "axiom dataset --help" },
    { id: "training", title: "Training planner and experiments", description: "Estimate hardware-aware plans and manage saved plan snapshots. Training execution is not implied by a plan.", scope: "Planning", availability: "Planning only", tone: "warning", icon: Hammer, page: "training", command: "axiom train --help" },
    { id: "workflows", title: "Workflow builder", description: "Create and validate graph designs, use templates, inspect revisions, duplicate designs, and run planning-only dry-runs.", scope: "Orchestration", availability: "Design only", tone: "warning", icon: GitBranch, page: "workflows", command: "axiom workflow --help" },
    { id: "evaluation", title: "Offline evaluation and regressions", description: "Score already-produced predictions, compare baselines, and review evaluation reports without model inference.", scope: "Quality gates", availability: "GUI + CLI", tone: "success", icon: CircleGauge, page: "evaluation", command: "axiom evaluation --help" },
    { id: "runtime", title: "Runtime connections", description: "Inspect runtime connection state. A GUI panel does not mean a local inference server is installed or running.", scope: "Operations", availability: "Connection-dependent", tone: "warning", icon: RadioTower, page: "runtime" },
    { id: "mcp", title: "MCP tools and permission profiles", description: "Inspect tool policy, named profiles, and supported local MCP surfaces. Tools remain deny-by-default until configured.", scope: "Integrations", availability: "GUI + CLI", tone: "success", icon: Network, page: "mcp", command: "axiom mcp --help" },
    { id: "diagnostics", title: "Diagnostics and support", description: "Check API and machine connectivity and access workspace warnings and support actions that are enabled.", scope: "Operations", availability: "GUI", tone: "success", icon: Wrench, page: "diagnostics", command: "axiom doctor --help" },
    { id: "api-playground", title: "API Playground", description: "Inspect a bounded allowlist of read-only HTTP endpoints and copy JSON responses.", scope: "Developer API", availability: "GUI · GET only", tone: "success", icon: Network, page: "api-playground" },
    { id: "agent-profiles", title: "Agent task profiles", description: "Version descriptive task labels without changing MCP grants or execution permissions.", scope: "Agent engineering", availability: "GUI + CLI metadata", tone: "success", icon: ShieldCheck, page: "agent-profiles", command: "axiom agent-profile --help" },
    { id: "logs", title: "Workspace event stream", description: "Filter recent workspace actions; this is not a claim of full OS or process telemetry.", scope: "Observability", availability: "GUI", tone: "success", icon: TerminalSquare, page: "logs" },
    { id: "settings", title: "Preferences and local security", description: "Manage available GUI preferences and administrator-session settings.", scope: "System", availability: "GUI", tone: "success", icon: Settings, page: "settings" },
    { id: "tools", title: "Developer tool catalog", description: "Discover supported external developer tools and review setup plans before explicit installs.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: Wrench, command: "axiom tools --help" },
    { id: "integrations", title: "Agent integration adapters", description: "Inspect configured local agent integration families without assuming a provider account is authenticated.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: Network, command: "axiom integration --help" },
    { id: "headroom", title: "Headroom context proxy integration", description: "Check the optional Headroom integration or its separate status/doctor commands; AXIOM does not silently start the proxy.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: Network, command: "axiom headroom status" },
    { id: "supercompress", title: "SuperCompress context utilities", description: "Review local context-compression commands. External endpoints or model calls require explicit configuration.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: SlidersHorizontal, command: "axiom supercompress --help" },
    { id: "ai-features", title: "AI helper commands and policy audit", description: "Access AI planning, Hugging Face search, and model policy-audit command families.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: BrainCircuit, command: "axiom ai --help" },
    { id: "huggingface", title: "Hugging Face access", description: "Manage CLI authentication, inspect account state, and invoke metadata/download actions only when explicitly requested.", scope: "Model providers", availability: "CLI surface", tone: "neutral", icon: Database, command: "axiom hf --help" },
    { id: "hardware", title: "System and hardware inspection", description: "Inspect OS/CPU/RAM/GPU facts used by hardware-aware planning, with estimated fit kept distinct from benchmark results.", scope: "System information", availability: "CLI surface", tone: "neutral", icon: Cpu, command: "axiom system info" },
    { id: "cli-optimizer", title: "CLI optimizer", description: "Run or inspect the terminal optimizer workflows and optimizer-profile commands.", scope: "CLI extensions", availability: "CLI surface", tone: "neutral", icon: Zap, command: "axiom optimizer --help" },
    { id: "project", title: "Project and configuration tools", description: "Initialize and validate local project structure, inspect config mappings, and review environment summaries.", scope: "CLI foundations", availability: "CLI surface", tone: "neutral", icon: FileCode2, command: "axiom project --help" },
    { id: "session", title: "Interactive CLI session tools", description: "Inspect the local CLI session-command family; the browser deliberately does not launch shell processes.", scope: "CLI foundations", availability: "CLI surface", tone: "neutral", icon: TerminalSquare, command: "axiom session --help" },
    { id: "api", title: "HTTP API reference", description: "Open the running service's OpenAPI explorer to inspect the endpoints actually exposed by this build.", scope: "Developer API", availability: "API docs", tone: "neutral", icon: Network, href: "/api/docs" },
    { id: "roadmap", title: "120-item product roadmap", description: "Inspect delivery labels from ROADMAP.md, search backlog items, and check ID/status integrity.", scope: "Governance", availability: "Source audit", tone: "neutral", icon: FileCode2, page: "roadmap", command: "axiom roadmap audit --json" },
  ];
  const queryTerms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = entries.filter((entry) => {
    const searchable = `${entry.title} ${entry.description} ${entry.scope} ${entry.availability} ${entry.command ?? ""}`.toLowerCase();
    return queryTerms.every((term) => searchable.includes(term));
  });
  const copyCommand = async (entry: Entry) => {
    if (!entry.command) return;
    try {
      await navigator.clipboard.writeText(entry.command);
      setCopiedCommand(entry.id);
    } catch {
      setCopiedCommand("");
    }
  };
  return <FeaturePage page="modules">
    <Surface title="Platform module directory" eyebrow="DISCOVER · OPEN · VERIFY" action={<StatusPill tone="neutral">{filtered.length} / {entries.length} modules</StatusPill>}>
      <p className="quiet-note">This directory maps product capabilities to their current entry points. “GUI” means a page exists; “CLI surface” means the current entry point is the terminal, not a fake in-browser executor. Planning-only and connection-dependent boundaries remain explicit.</p>
      <label className="event-filter"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search modules, capabilities, or commands" aria-label="Search AXIOM modules" /></label>
      <div className="module-list">
        {filtered.map((entry) => {
          const Icon = entry.icon;
          return <div className="module-row" key={entry.id} role="group" aria-label={entry.title}>
            <span className="module-icon"><Icon size={17} /></span>
            <span className="module-copy"><b>{entry.title}</b><small>{entry.description}</small><small>{entry.scope}{entry.command ? ` · ${entry.command}` : ""}</small></span>
            <StatusPill tone={entry.tone}>{entry.availability}</StatusPill>
            <span className="module-row-actions">
              {entry.page && <button type="button" className="button secondary-button" onClick={() => setPage(entry.page!)}>Open</button>}
              {entry.href && <a className="button secondary-button" href={entry.href} target="_blank" rel="noreferrer">Open API</a>}
              {entry.command && !entry.page && <button type="button" className="button secondary-button" onClick={() => void copyCommand(entry)}>{copiedCommand === entry.id ? "Copied" : "Copy command"}</button>}
            </span>
          </div>;
        })}
      </div>
      {filtered.length === 0 && <EmptyState icon={Search} title="No modules match" description="Try another capability name or command family." />}
    </Surface>
    <Callout tone="info" title="Feature boundaries are intentional">The directory improves discoverability; it does not mark planned roadmap items as implemented or create execution support that is not in the backend.</Callout>
  </FeaturePage>;
}

function RoadmapPage() {
  const [report, setReport] = useState<RoadmapAudit | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<"all" | "open" | "implemented">("all");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      setReport(await axiomApi.roadmapAudit());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load the roadmap audit.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);
  const normalizedQuery = query.trim().toLowerCase();
  const visibleItems = (report?.items ?? []).filter((item) => {
    const matchesQuery = !normalizedQuery || `${item.id} ${item.title} ${item.group} ${item.status_label}`.toLowerCase().includes(normalizedQuery);
    const matchesStatus = filter === "all" || (filter === "implemented" ? item.status === "implemented" : item.status !== "implemented");
    return matchesQuery && matchesStatus;
  });

  return <FeaturePage page="roadmap" actions={<button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}><RefreshCcw size={16} /> {loading ? "Auditing…" : "Refresh roadmap"}</button>}>
    {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
    {report && <div className="summary-grid">
      <SummaryCard icon={Check} label="Implemented" value={String(report.counts.implemented)} detail="Source roadmap marks these as tested" tone={report.counts.implemented === 120 ? "success" : "neutral"} />
      <SummaryCard icon={Clock3} label="In progress" value={String(report.counts.in_progress)} detail="Validation or implementation remains" tone={report.counts.in_progress ? "warning" : "success"} />
      <SummaryCard icon={CircleAlert} label="Planned" value={String(report.counts.planned)} detail="Not marked as delivered" tone={report.counts.planned ? "warning" : "success"} />
      <SummaryCard icon={FileCode2} label="Scope integrity" value={report.integrity.ok ? `${report.total}/120` : "REVIEW"} detail={report.integrity.ok ? "IDs are complete and unique" : "Missing, duplicate, or unknown roadmap entries"} tone={report.integrity.ok ? "success" : "danger"} />
    </div>}
    {report && <Surface title="Scoped product roadmap" eyebrow="SOURCE-OF-TRUTH STATUS">
      <p className="quiet-note">Statuses are read from ROADMAP.md and are not inferred from file presence. {report.counts.implemented} of {report.total} entries are currently marked “Implemented; tests passing”.</p>
      {!report.integrity.ok && <Callout tone="warning" title="Roadmap integrity needs review">Missing IDs: {report.integrity.missing_ids.join(", ") || "none"}. Duplicates: {report.integrity.duplicate_ids.join(", ") || "none"}. Unknown statuses: {report.integrity.unknown_status_ids.join(", ") || "none"}.</Callout>}
      <label className="event-filter"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter by ID, title, group, or status" aria-label="Search roadmap items" /></label>
      <div className="segmented-control" role="group" aria-label="Roadmap status filter">
        <button type="button" className={filter === "all" ? "active" : ""} onClick={() => setFilter("all")}>All</button>
        <button type="button" className={filter === "open" ? "active" : ""} onClick={() => setFilter("open")}>Open work</button>
        <button type="button" className={filter === "implemented" ? "active" : ""} onClick={() => setFilter("implemented")}>Implemented</button>
      </div>
      {loading ? <div className="loading-state" role="status">Reading ROADMAP.md…</div> : <div className="event-list">{visibleItems.map((item) => {
        const tone: Tone = item.status === "implemented" ? "success" : item.status === "in_progress" ? "warning" : item.status === "unknown" ? "danger" : "neutral";
        return <div className="event-row" key={item.id}><span className="event-marker" /><div><b>{item.id} · {item.title}</b><small>{item.group} · {item.status_label}</small></div><StatusPill tone={tone}>{item.status_label.toUpperCase()}</StatusPill></div>;
      })}{visibleItems.length === 0 && <EmptyState icon={Search} title="No roadmap items match" description="Change the search or status filter to see more entries." />}</div>}
      <p className="quiet-note">Showing {visibleItems.length} of {report.total} source entries · generated {formatTimestamp(report.generated_at)}.</p>
    </Surface>}
  </FeaturePage>;
}

function LogsPage() {
  const [events, setEvents] = useState<WorkspaceEvent[]>([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      setEvents(await axiomApi.events());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load workspace events.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);
  const normalizedFilter = filter.trim().toLowerCase();
  const visibleEvents = events.filter((event) => `${event.action} ${event.detail || ""}`.toLowerCase().includes(normalizedFilter));

  return <FeaturePage page="logs" actions={<button type="button" className="button secondary-button" onClick={() => void refresh()} disabled={loading}><RefreshCcw size={16} /> {loading ? "Refreshing" : "Refresh"}</button>}>
    <Surface title="Workspace event stream" eyebrow="LOCAL / RECENT" action={<StatusPill tone={events.length ? "success" : "neutral"}>{events.length} EVENTS</StatusPill>}>
      <label className="event-filter"><Search size={15} /><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter by action or detail" aria-label="Filter workspace events" /></label>
      {error && <InlineAlert tone="warning" role="alert">{error}</InlineAlert>}
      {loading ? <div className="loading-state" role="status">Loading local events…</div> : visibleEvents.length ? <div className="event-list">{visibleEvents.map((event, index) => <div className="event-row" key={`${event.timestamp}-${index}`}><span className="event-marker" /><div><b>{event.action.replaceAll(".", " / ")}</b><small>{event.detail || "Completed"} · {formatTimestamp(event.timestamp)}</small></div></div>)}</div> : <EmptyState icon={TerminalSquare} title={events.length ? "No events match" : "No workspace events yet"} description={events.length ? "Try a different filter." : "Successful model registration, dataset profiling, and training plans appear here."} />}
    </Surface>
    <div className="log-contract"><span><FileCode2 size={15} /> Event retention</span><code>Recent local actions · in memory · bounded</code></div>
  </FeaturePage>;
}

function SettingsPage({
  theme,
  setTheme,
  onChangePassword,
}: {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  onChangePassword: () => void;
}) {
  return <FeaturePage page="settings">
    <div className="settings-grid">
      <Surface title="Appearance" eyebrow="INTERFACE">
        <div className="setting-row"><div><b>Theme</b><span>Choose how AXIOM appears on this device.</span></div><div className="segmented-control" role="group" aria-label="Theme"><button type="button" className={theme === "dark" ? "active" : ""} onClick={() => setTheme("dark")}><Moon size={15} /> Dark</button><button type="button" className={theme === "light" ? "active" : ""} onClick={() => setTheme("light")}><Sun size={15} /> Light</button></div></div>
        <div className="surface-divider" />
        <div className="setting-row"><div><b>Command search</b><span>Use the keyboard shortcut to jump between workspace views.</span></div><kbd className="shortcut-key"><Command size={12} /> K</kbd></div>
      </Surface>
      <Surface title="Security" eyebrow="LOCAL ADMINISTRATOR">
        <div className="setting-row"><div><b>Administrator password</b><span>Changing it ends the current session and returns to sign in.</span></div><button type="button" className="button secondary-button" onClick={onChangePassword}><KeyRound size={15} /> Change password</button></div>
        <div className="surface-divider" />
        <div className="setting-row"><div><b>Session timeout</b><span>Automatic sign out after inactivity.</span></div><StatusPill tone="neutral">10 MINUTES</StatusPill></div>
      </Surface>
      <Surface title="Integration boundary" eyebrow="HONEST STATE">
        <Callout tone="info" title="Local API integration">The dashboard, model registry, dataset profiler, training planner, hardware snapshot, persistent plan history, diagnostics, and event log share the local FastAPI contract.</Callout>
      </Surface>
    </div>
  </FeaturePage>;
}

function Surface({ title, eyebrow, action, children, className = "" }: { title: string; eyebrow?: string; action?: ReactNode; children: ReactNode; className?: string }) {
  return <section className={`surface ${className}`}><div className="surface-head"><div>{eyebrow && <p className="surface-eyebrow">{eyebrow}</p>}<h2>{title}</h2></div>{action && <div className="surface-action">{action}</div>}</div>{children}</section>;
}

function InfoCard({ icon: Icon, label, value, detail, tone }: { icon: LucideIcon; label: string; value: string; detail: string; tone: Tone }) {
  return <div className="info-card"><div className="info-card-head"><span>{label}</span><Icon size={17} /></div><strong>{value}</strong><StatusPill tone={tone}>{detail}</StatusPill></div>;
}

function EmptyState({ icon: Icon, title, description, action }: { icon: LucideIcon; title: string; description: string; action?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={20} /></div><h3>{title}</h3><p>{description}</p>{action && <div className="empty-action">{action}</div>}</div>;
}

function StatusPill({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`status-pill ${tone}`}><i />{children}</span>;
}

function InlineAlert({ tone, children, role = "status" }: { tone: Tone; children: ReactNode; role?: "status" | "alert" }) {
  return <div className={`inline-alert ${tone}`} role={role}><CircleAlert size={15} />{children}</div>;
}

function Callout({ tone, title, children }: { tone: Tone; title: string; children: ReactNode }) {
  return <div className={`callout ${tone}`}><div className="callout-icon">{tone === "warning" ? <TriangleAlert size={16} /> : <CircleAlert size={16} />}</div><div><b>{title}</b><p>{children}</p></div></div>;
}

function CliReference({ command, disabled = false }: { command: string; disabled?: boolean }) {
  return <button type="button" className="cli-reference" disabled={disabled} onClick={() => navigator.clipboard?.writeText(command)}><TerminalSquare size={15} /><code>{command}</code><span>{disabled ? "Unavailable" : "Copy"}</span></button>;
}

function DiagnosticRow({ icon: Icon, title, detail, state, tone }: { icon: LucideIcon; title: string; detail: string; state: string; tone: Tone }) {
  return <div className="diagnostic-row"><span className={`diagnostic-icon ${tone}`}><Icon size={16} /></span><div><b>{title}</b><span>{detail}</span></div><StatusPill tone={tone}>{state}</StatusPill></div>;
}

function ChangePasswordDialog({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNext, setShowNext] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);
  const checks = getPasswordChecks(next);
  const complete = Object.values(checks).every(Boolean) && next.length > 0 && next === confirm;

  useEffect(() => {
    closeRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, saving]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSaving(true);

    try {
      const validCurrent = await verifyAdministrator(getAdministratorName(), current);
      if (!validCurrent) {
        setError("Current password is incorrect.");
        return;
      }
      if (!complete) {
        setError("The new password does not meet all requirements.");
        return;
      }
      await saveAdministrator(getAdministratorName(), next);
      setSuccess(true);
      window.setTimeout(onSuccess, 1300);
    } catch {
      setError("Unable to change the administrator password.");
    } finally {
      setSaving(false);
    }
  }

  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !saving) onClose(); }}>
    <section className="dialog" role="dialog" aria-modal="true" aria-labelledby="password-dialog-title" aria-describedby="password-dialog-description" onMouseDown={(event) => event.stopPropagation()}>
      {success ? <div className="dialog-success"><div className="success-mark"><Check size={24} /></div><p className="eyebrow">SECURITY / UPDATED</p><h2>Password changed.</h2><p id="password-dialog-description">Returning to the AXIOM sign in screen...</p></div> : <form onSubmit={submit}>
        <div className="dialog-head"><div><p className="eyebrow">SECURITY / LOCAL ADMINISTRATOR</p><h2 id="password-dialog-title">Change password</h2></div><button ref={closeRef} type="button" className="icon-button" aria-label="Close change password dialog" onClick={onClose}><X size={17} /></button></div>
        <p id="password-dialog-description" className="form-intro">Changing the password ends the current session after the update.</p>
        <Field label="Current password" htmlFor="current-password" action={<button type="button" className="field-action" aria-pressed={showCurrent} onClick={() => setShowCurrent((value) => !value)}>{showCurrent ? "Hide" : "Show"}</button>}><input id="current-password" type={showCurrent ? "text" : "password"} value={current} onChange={(event) => setCurrent(event.target.value)} autoComplete="current-password" /></Field>
        <Field label="New password" htmlFor="new-password" action={<button type="button" className="field-action" aria-pressed={showNext} onClick={() => setShowNext((value) => !value)}>{showNext ? "Hide" : "Show"}</button>}><input id="new-password" type={showNext ? "text" : "password"} value={next} onChange={(event) => setNext(event.target.value)} autoComplete="new-password" /></Field>
        <div className="requirements"><Requirement ok={checks.length} label="8+ characters" /><Requirement ok={checks.upper} label="Uppercase letter" /><Requirement ok={checks.lower} label="Lowercase letter" /><Requirement ok={checks.number} label="Contains a number" /></div>
        <Field label="Confirm new password" htmlFor="confirm-password" hint={confirm.length > 0 ? (next === confirm ? "Passwords match" : "Passwords do not match") : undefined} hintTone={next === confirm && confirm.length > 0 ? "success" : "default"}><input id="confirm-password" type="password" value={confirm} onChange={(event) => setConfirm(event.target.value)} autoComplete="new-password" /></Field>
        {error && <InlineAlert tone="danger" role="alert">{error}</InlineAlert>}
        <button type="submit" className="button primary-button full-width" disabled={!complete || saving}>{saving ? "Changing password..." : "Change password"}<ArrowRight size={17} /></button>
      </form>}
    </section>
  </div>;
}

function NoticeToast({ notice, onDismiss }: { notice: Notice; onDismiss: () => void }) {
  return <div className={`notice-toast ${notice.tone}`} role="status"><span className="notice-icon">{notice.tone === "success" ? <CircleCheck size={16} /> : notice.tone === "warning" ? <TriangleAlert size={16} /> : <CircleAlert size={16} />}</span><span>{notice.message}</span><button type="button" className="toast-close" aria-label="Dismiss notification" onClick={onDismiss}><X size={15} /></button></div>;
}

export default App;
