import {
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";

import {
  changePassword,
  createAdministrator,
  getAdministratorName,
  getSession,
  hasAdministrator,
  login,
  logout,
  passwordRequirements,
  type PasswordRequirements,
} from "./auth";

type AuthScreen =
  | "firstboot"
  | "login"
  | "app";

type AuthScreensProps = {
  onAuthenticated: () => void;
  onLoggedOut: () => void;
};

function RequirementList({
  requirements,
}: {
  requirements: PasswordRequirements;
}) {
  const items = [
    ["length", "At least 12 characters"],
    ["uppercase", "One uppercase letter"],
    ["lowercase", "One lowercase letter"],
    ["number", "One number"],
    ["special", "One special character"],
  ] as const;

  return (
    <div className="axiom-password-requirements">
      {items.map(([key, label]) => {
        const valid = requirements[key];

        return (
          <div
            key={key}
            className={`axiom-password-requirement ${
              valid ? "valid" : ""
            }`}
          >
            <span className="axiom-requirement-light">
              {valid ? "✓" : ""}
            </span>

            <span>{label}</span>
          </div>
        );
      })}
    </div>
  );
}

function PasswordField({
  value,
  onChange,
  placeholder,
  autoComplete,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  autoComplete: string;
}) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="axiom-password-field">
      <input
        type={visible ? "text" : "password"}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        autoComplete={autoComplete}
      />

      <button
        type="button"
        className="axiom-password-toggle"
        onClick={() => setVisible((current) => !current)}
      >
        {visible ? "HIDE" : "SHOW"}
      </button>
    </div>
  );
}

function Brand() {
  return (
    <div className="axiom-auth-brand">
      <div className="axiom-auth-mark">A</div>

      <div>
        <strong>AXIOM</strong>
        <span>AI ENGINEERING PLATFORM</span>
      </div>
    </div>
  );
}

function FirstBoot({
  onAuthenticated,
}: {
  onAuthenticated: () => void;
}) {
  const [username, setUsername] = useState("Administrator");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [complete, setComplete] = useState(false);

  const requirements = useMemo(
    () => passwordRequirements(password),
    [password],
  );

  const matches = password.length > 0 && password === confirm;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");

    try {
      await createAdministrator(username, password);

      setComplete(true);

      window.setTimeout(() => {
        onAuthenticated();
      }, 1300);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to create administrator.",
      );
    }
  }

  if (complete) {
    return (
      <div className="axiom-auth-screen">
        <div className="axiom-auth-success">
          <div className="axiom-success-orb">
            ✓
          </div>

          <span className="axiom-auth-kicker">
            FIRST BOOT COMPLETE
          </span>

          <h1>Administrator created.</h1>

          <p>
            Your AXIOM Control Center is ready.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="axiom-auth-screen">
      <div className="axiom-auth-layout">
        <div className="axiom-auth-copy">
          <Brand />

          <span className="axiom-auth-kicker">
            FIRST BOOT • 01 / 03
          </span>

          <h1>
            Build your
            <span> control center.</span>
          </h1>

          <p>
            Create the local administrator account that protects
            this AXIOM workspace.
          </p>

          <div className="axiom-auth-points">
            <div>
              <span>01</span>
              <p>Local administrator access</p>
            </div>

            <div>
              <span>02</span>
              <p>Protected Control Center session</p>
            </div>

            <div>
              <span>03</span>
              <p>Ready for AXIOM platform services</p>
            </div>
          </div>
        </div>

        <form
          className="axiom-auth-card"
          onSubmit={submit}
        >
          <div className="axiom-card-kicker">
            ADMINISTRATOR
          </div>

          <h2>Create administrator</h2>

          <p>
            This account will be used to access your local
            AXIOM Control Center.
          </p>

          <label>
            Administrator name
            <input
              value={username}
              onChange={(event) =>
                setUsername(event.target.value)
              }
              autoComplete="username"
            />
          </label>

          <label>
            Password
            <PasswordField
              value={password}
              onChange={setPassword}
              placeholder="Create a strong password"
              autoComplete="new-password"
            />
          </label>

          <RequirementList
            requirements={requirements}
          />

          <label>
            Confirm password
            <PasswordField
              value={confirm}
              onChange={setConfirm}
              placeholder="Repeat your password"
              autoComplete="new-password"
            />
          </label>

          {confirm.length > 0 && (
            <div
              className={`axiom-password-match ${
                matches ? "valid" : ""
              }`}
            >
              {matches
                ? "Passwords match"
                : "Passwords do not match"}
            </div>
          )}

          {error && (
            <div className="axiom-auth-error">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="axiom-auth-primary"
            disabled={
              !username.trim() ||
              !Object.values(requirements).every(Boolean) ||
              !matches
            }
          >
            Create Administrator
          </button>

          <small className="axiom-auth-security">
            Local administrator setup
          </small>
        </form>
      </div>
    </div>
  );
}

function Login({
  onAuthenticated,
}: {
  onAuthenticated: () => void;
}) {
  const [username, setUsername] = useState(
    getAdministratorName(),
  );

  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loggingIn, setLoggingIn] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();

    setError("");
    setLoggingIn(true);

    try {
      const authenticated = await login(
        username.trim(),
        password,
      );

      if (!authenticated) {
        setError(
          "Administrator name or password is incorrect.",
        );
        return;
      }

      onAuthenticated();
    } finally {
      setLoggingIn(false);
    }
  }

  return (
    <div className="axiom-auth-screen axiom-login-screen">
      <div className="axiom-login-glow axiom-login-glow-red" />
      <div className="axiom-login-glow axiom-login-glow-blue" />

      <form
        className="axiom-login-card"
        onSubmit={submit}
      >
        <Brand />

        <span className="axiom-auth-kicker">
          AXIOM CONTROL CENTER
        </span>

        <h1>Welcome back.</h1>

        <p>
          Authenticate to continue to your local AXIOM workspace.
        </p>

        <label>
          Administrator
          <input
            value={username}
            onChange={(event) =>
              setUsername(event.target.value)
            }
            autoComplete="username"
          />
        </label>

        <label>
          Password
          <PasswordField
            value={password}
            onChange={setPassword}
            placeholder="Enter your password"
            autoComplete="current-password"
          />
        </label>

        {error && (
          <div className="axiom-auth-error">
            {error}
          </div>
        )}

        <button
          type="submit"
          className="axiom-auth-primary"
          disabled={loggingIn}
        >
          {loggingIn ? "Authenticating..." : "Sign in"}
        </button>

        <small className="axiom-auth-security">
          Local AXIOM authentication
        </small>
      </form>
    </div>
  );
}

function ChangePassword({
  onClose,
  onChanged,
}: {
  onClose: () => void;
  onChanged: () => void;
}) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const requirements = useMemo(
    () => passwordRequirements(next),
    [next],
  );

  const complete =
    Object.values(requirements).every(Boolean) &&
    next === confirm &&
    next.length > 0;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");

    try {
      await changePassword(current, next);

      setSuccess(true);

      window.setTimeout(() => {
        logout();
        onChanged();
      }, 1500);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to change password.",
      );
    }
  }

  if (success) {
    return (
      <div className="axiom-modal-backdrop">
        <div className="axiom-password-success">
          <div className="axiom-success-orb">
            ✓
          </div>

          <span>SECURITY</span>

          <h2>Password changed.</h2>

          <p>
            Returning you to the AXIOM sign-in screen.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="axiom-modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <form
        className="axiom-change-password"
        onSubmit={submit}
      >
        <div className="axiom-modal-header">
          <div>
            <span>SECURITY</span>
            <h2>Change password</h2>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="axiom-modal-close"
          >
            ×
          </button>
        </div>

        <label>
          Current password
          <PasswordField
            value={current}
            onChange={setCurrent}
            placeholder="Current password"
            autoComplete="current-password"
          />
        </label>

        <label>
          New password
          <PasswordField
            value={next}
            onChange={setNext}
            placeholder="Create a new password"
            autoComplete="new-password"
          />
        </label>

        <RequirementList
          requirements={requirements}
        />

        <label>
          Confirm new password
          <PasswordField
            value={confirm}
            onChange={setConfirm}
            placeholder="Repeat the new password"
            autoComplete="new-password"
          />
        </label>

        {confirm.length > 0 && (
          <div
            className={`axiom-password-match ${
              next === confirm ? "valid" : ""
            }`}
          >
            {next === confirm
              ? "Passwords match"
              : "Passwords do not match"}
          </div>
        )}

        {error && (
          <div className="axiom-auth-error">
            {error}
          </div>
        )}

        <button
          type="submit"
          className="axiom-auth-primary"
          disabled={!complete}
        >
          Change password
        </button>
      </form>
    </div>
  );
}

export function AuthScreens({
  onAuthenticated,
  onLoggedOut,
}: AuthScreensProps) {
  const [screen, setScreen] = useState<AuthScreen>(
    hasAdministrator() && getSession()
      ? "app"
      : hasAdministrator()
        ? "login"
        : "firstboot",
  );

  const [menuOpen, setMenuOpen] = useState(false);
  const [changeOpen, setChangeOpen] = useState(false);

  useEffect(() => {
    if (screen !== "app") {
      return;
    }

    let timer: number | undefined;

    const resetTimer = () => {
      if (timer) {
        window.clearTimeout(timer);
      }

      timer = window.setTimeout(() => {
        logout();

        setMenuOpen(false);
        setChangeOpen(false);
        setScreen("login");

        onLoggedOut();
      }, 10 * 60 * 1000);
    };

    const events = [
      "mousemove",
      "mousedown",
      "keydown",
      "touchstart",
      "scroll",
      "pointerdown",
    ];

    events.forEach((event) =>
      window.addEventListener(event, resetTimer),
    );

    resetTimer();

    return () => {
      if (timer) {
        window.clearTimeout(timer);
      }

      events.forEach((event) =>
        window.removeEventListener(event, resetTimer),
      );
    };
  }, [screen, onLoggedOut]);

  if (screen === "firstboot") {
    return (
      <FirstBoot
        onAuthenticated={() => {
          setScreen("login");
        }}
      />
    );
  }

  if (screen === "login") {
    return (
      <Login
        onAuthenticated={() => {
          setScreen("app");
          onAuthenticated();
        }}
      />
    );
  }

  return (
    <>
      <div className="axiom-auth-user-area">
        <button
          className="axiom-user-trigger"
          onClick={() =>
            setMenuOpen((current) => !current)
          }
        >
          <span className="axiom-user-avatar">
            {getAdministratorName()
              .charAt(0)
              .toUpperCase()}
          </span>

          <span className="axiom-user-name">
            {getAdministratorName()}
          </span>

          <span className="axiom-user-chevron">
            {menuOpen ? "⌃" : "⌄"}
          </span>
        </button>

        {menuOpen && (
          <div className="axiom-user-menu">
            <button
              onClick={() => {
                setChangeOpen(true);
                setMenuOpen(false);
              }}
            >
              Change password
            </button>

            <button
              className="danger"
              onClick={() => {
                logout();
                setMenuOpen(false);
                setScreen("login");
                onLoggedOut();
              }}
            >
              Logout
            </button>
          </div>
        )}
      </div>

      {changeOpen && (
        <ChangePassword
          onClose={() => setChangeOpen(false)}
          onChanged={() => {
            setChangeOpen(false);
            setScreen("login");
            onLoggedOut();
          }}
        />
      )}
    </>
  );
}
