const AUTH_KEY = "axiom.auth";
const SESSION_KEY = "axiom.session";

export type PasswordRequirements = {
  length: boolean;
  uppercase: boolean;
  lowercase: boolean;
  number: boolean;
  special: boolean;
};

export type AuthRecord = {
  username: string;
  passwordHash: string;
  createdAt: number;
  updatedAt: number;
};

export type Session = {
  username: string;
  loggedInAt: number;
};

async function hashPassword(password: string): Promise<string> {
  const data = new TextEncoder().encode(password);
  const digest = await crypto.subtle.digest("SHA-256", data);

  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function passwordRequirements(
  password: string,
): PasswordRequirements {
  return {
    length: password.length >= 12,
    uppercase: /[A-Z]/.test(password),
    lowercase: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[^A-Za-z0-9]/.test(password),
  };
}

export function passwordIsValid(password: string): boolean {
  const requirements = passwordRequirements(password);

  return Object.values(requirements).every(Boolean);
}

export function hasAdministrator(): boolean {
  return Boolean(localStorage.getItem(AUTH_KEY));
}

export async function createAdministrator(
  username: string,
  password: string,
): Promise<void> {
  if (!username.trim()) {
    throw new Error("Administrator name is required.");
  }

  if (!passwordIsValid(password)) {
    throw new Error("Password does not satisfy all requirements.");
  }

  const record: AuthRecord = {
    username: username.trim(),
    passwordHash: await hashPassword(password),
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };

  localStorage.setItem(AUTH_KEY, JSON.stringify(record));
}

export async function login(
  username: string,
  password: string,
): Promise<boolean> {
  const raw = localStorage.getItem(AUTH_KEY);

  if (!raw) {
    return false;
  }

  const record = JSON.parse(raw) as AuthRecord;

  const passwordHash = await hashPassword(password);

  if (
    record.username !== username ||
    record.passwordHash !== passwordHash
  ) {
    return false;
  }

  const session: Session = {
    username: record.username,
    loggedInAt: Date.now(),
  };

  sessionStorage.setItem(
    SESSION_KEY,
    JSON.stringify(session),
  );

  return true;
}

export function getSession(): Session | null {
  const raw = sessionStorage.getItem(SESSION_KEY);

  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as Session;
  } catch {
    return null;
  }
}

export function logout(): void {
  sessionStorage.removeItem(SESSION_KEY);
}

export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const raw = localStorage.getItem(AUTH_KEY);

  if (!raw) {
    throw new Error("Administrator account does not exist.");
  }

  const record = JSON.parse(raw) as AuthRecord;

  const currentHash = await hashPassword(currentPassword);

  if (currentHash !== record.passwordHash) {
    throw new Error("Current password is incorrect.");
  }

  if (!passwordIsValid(newPassword)) {
    throw new Error("New password does not satisfy all requirements.");
  }

  record.passwordHash = await hashPassword(newPassword);
  record.updatedAt = Date.now();

  localStorage.setItem(
    AUTH_KEY,
    JSON.stringify(record),
  );
}

export function getAdministratorName(): string {
  const session = getSession();

  if (session) {
    return session.username;
  }

  const raw = localStorage.getItem(AUTH_KEY);

  if (!raw) {
    return "Administrator";
  }

  try {
    return (JSON.parse(raw) as AuthRecord).username;
  } catch {
    return "Administrator";
  }
}
