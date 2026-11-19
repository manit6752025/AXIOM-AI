const DB_NAME = "axiom-secure-storage-v1";
const KEY_STORE = "keys";
const RECORD_STORE = "records";
const KEY_ID = "administrator-key-v1";
const RECORD_ID = "administrator-record-v1";

interface StoredEncryptedRecord {
  id: string;
  iv: number[];
  ciphertext: number[];
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(KEY_STORE)) db.createObjectStore(KEY_STORE, { keyPath: "id" });
      if (!db.objectStoreNames.contains(RECORD_STORE)) db.createObjectStore(RECORD_STORE, { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Unable to open secure storage"));
  });
}

async function getKey(): Promise<CryptoKey> {
  const db = await openDatabase();
  const existing = await new Promise<{ id: string; key: CryptoKey } | undefined>((resolve, reject) => {
    const request = db.transaction(KEY_STORE, "readonly").objectStore(KEY_STORE).get(KEY_ID);
    request.onsuccess = () => resolve(request.result as { id: string; key: CryptoKey } | undefined);
    request.onerror = () => reject(request.error ?? new Error("Unable to read secure storage key"));
  });
  if (existing?.key) {
    db.close();
    return existing.key;
  }

  const key = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );

  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(KEY_STORE, "readwrite");
    tx.objectStore(KEY_STORE).put({ id: KEY_ID, key });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("Unable to store secure storage key"));
  });
  db.close();
  return key;
}

async function setEncryptedRecord(record: unknown): Promise<void> {
  const key = await getKey();
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const plaintext = new TextEncoder().encode(JSON.stringify(record));
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, plaintext),
  );
  const payload: StoredEncryptedRecord = {
    id: RECORD_ID,
    iv: Array.from(iv),
    ciphertext: Array.from(ciphertext),
  };
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(RECORD_STORE, "readwrite");
    tx.objectStore(RECORD_STORE).put(payload);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("Unable to store encrypted administrator record"));
  });
  db.close();
}

async function getEncryptedRecord(): Promise<string | null> {
  const key = await getKey();
  const db = await openDatabase();
  const payload = await new Promise<StoredEncryptedRecord | undefined>((resolve, reject) => {
    const request = db.transaction(RECORD_STORE, "readonly").objectStore(RECORD_STORE).get(RECORD_ID);
    request.onsuccess = () => resolve(request.result as StoredEncryptedRecord | undefined);
    request.onerror = () => reject(request.error ?? new Error("Unable to read encrypted administrator record"));
  });
  db.close();
  if (!payload) return null;

  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: new Uint8Array(payload.iv) },
    key,
    new Uint8Array(payload.ciphertext),
  );
  return new TextDecoder().decode(plaintext);
}

async function removeEncryptedRecord(): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(RECORD_STORE, "readwrite");
    tx.objectStore(RECORD_STORE).delete(RECORD_ID);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("Unable to delete administrator record"));
  });
  db.close();
}

export const secureAdministratorStore = {
  set: setEncryptedRecord,
  get: getEncryptedRecord,
  remove: removeEncryptedRecord,
};
