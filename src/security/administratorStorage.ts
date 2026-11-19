export interface AdministratorVerifierRecord {
  username: string;
  salt: string;
  iv: string;
  ciphertext: string;
}

const DB_NAME = 'axiom-security';
const STORE_NAME = 'administrator';
const RECORD_KEY = 'current';

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Unable to open security storage'));
  });
}

export async function saveAdministratorVerifier(
  record: AdministratorVerifierRecord,
): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(record, RECORD_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Unable to save administrator verifier'));
  });
  db.close();
}

export async function readAdministratorVerifier(): Promise<AdministratorVerifierRecord | null> {
  const db = await openDatabase();
  const result = await new Promise<AdministratorVerifierRecord | null>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const request = tx.objectStore(STORE_NAME).get(RECORD_KEY);
    request.onsuccess = () => resolve((request.result ?? null) as AdministratorVerifierRecord | null);
    request.onerror = () => reject(request.error ?? new Error('Unable to read administrator verifier'));
  });
  db.close();
  return result;
}

export async function clearAdministratorVerifier(): Promise<void> {
  const db = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete(RECORD_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('Unable to clear administrator verifier'));
  });
  db.close();
}
