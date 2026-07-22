/**
 * Chart notes backed by Firebase (Firestore + Auth).
 *
 * Collections
 *   notes/{id}       { runId, specId, context, step, uid, handle, text, createdAt }
 *   handles/{uid}    { number, handle, createdAt }   – locks a handle to a browser
 *   meta/counter     { next }                        – allocates the next number
 *
 * Identity
 *   Every visitor is signed in with Firebase Anonymous Auth. Firebase keeps the
 *   same uid in this browser across sessions, so their handle (explorer_001) is
 *   stable and reused on every curve. The admin signs in with email/password.
 */
const NotesStore = (() => {
  let app = null;
  let auth = null;
  let db = null;

  let readyPromise = null;
  let backendReady = false;
  let authReadyPromise = null;
  let authReadyResolve = null;
  let lastAuthError = null;

  let myUid = null;
  let myHandle = null;
  let adminMode = false;

  const authListeners = new Set();

  function resetAuthReady() {
    authReadyPromise = new Promise((resolve) => {
      authReadyResolve = resolve;
    });
  }

  function markAuthReady() {
    if (authReadyResolve) {
      authReadyResolve();
      authReadyResolve = null;
    }
  }

  async function waitForUid(timeoutMs = 8000) {
    if (myUid) return myUid;
    await init();
    if (myUid) return myUid;
    await Promise.race([
      authReadyPromise || Promise.resolve(),
      new Promise((r) => setTimeout(r, timeoutMs)),
    ]);
    if (myUid) return myUid;
    if (lastAuthError) {
      throw new Error(
        "Anonymous sign-in failed. Enable Anonymous in Firebase Authentication → Sign-in method. (" +
          lastAuthError +
          ")"
      );
    }
    throw new Error(
      "Still signing in. Check that Anonymous auth is enabled in Firebase, then refresh the page."
    );
  }

  function cfg() {
    return window.NOTES_CONFIG || {};
  }
  function fb() {
    return cfg().firebase || {};
  }
  function adminUid() {
    return cfg().adminUid || "";
  }

  function configured() {
    const f = fb();
    return Boolean(
      f.apiKey && f.projectId && f.appId && typeof firebase !== "undefined"
    );
  }

  function padHandle(n) {
    const prefix = cfg().handlePrefix || "explorer_";
    const pad = Number(cfg().handlePad) || 3;
    return prefix + String(n).padStart(pad, "0");
  }

  function emitAuth() {
    const state = {
      ready: backendReady,
      uid: myUid,
      handle: adminMode ? "admin" : myHandle,
      isAdmin: adminMode,
    };
    authListeners.forEach((fn) => {
      try {
        fn(state);
      } catch (_) {}
    });
  }

  function onAuthChange(fn) {
    authListeners.add(fn);
    // Fire immediately with current state.
    fn({ ready: backendReady, uid: myUid, handle: adminMode ? "admin" : myHandle, isAdmin: adminMode });
    return () => authListeners.delete(fn);
  }

  /** Allocate (or reuse) this browser's stable handle via a counter transaction. */
  async function ensureHandle(uid) {
    const handleRef = db.collection("handles").doc(uid);
    const existing = await handleRef.get();
    if (existing.exists && existing.data().handle) {
      return existing.data().handle;
    }
    const counterRef = db.collection("meta").doc("counter");
    return db.runTransaction(async (tx) => {
      // Re-check inside the transaction to avoid double allocation.
      const hSnap = await tx.get(handleRef);
      if (hSnap.exists && hSnap.data().handle) return hSnap.data().handle;
      const cSnap = await tx.get(counterRef);
      const next = (cSnap.exists ? Number(cSnap.data().next) : 1) || 1;
      const handle = padHandle(next);
      tx.set(counterRef, { next: next + 1 }, { merge: true });
      tx.set(handleRef, {
        number: next,
        handle,
        createdAt: firebase.firestore.FieldValue.serverTimestamp(),
      });
      return handle;
    });
  }

  async function handleAuthedUser(user) {
    myUid = user.uid;
    lastAuthError = null;
    if (user.uid === adminUid() && !user.isAnonymous) {
      adminMode = true;
      myHandle = "admin";
    } else {
      adminMode = false;
      try {
        myHandle = await ensureHandle(user.uid);
      } catch (err) {
        console.warn("Handle allocation failed", err);
        myHandle = null;
      }
    }
    emitAuth();
    markAuthReady();
  }

  async function init() {
    if (readyPromise) return readyPromise;
    resetAuthReady();
    readyPromise = (async () => {
      if (!configured()) {
        backendReady = false;
        lastAuthError = "Firebase is not configured";
        emitAuth();
        markAuthReady();
        return;
      }
      app = firebase.apps?.length ? firebase.app() : firebase.initializeApp(fb());
      auth = firebase.auth();
      db = firebase.firestore();
      try {
        await auth.setPersistence(firebase.auth.Auth.Persistence.LOCAL);
      } catch (_) {}

      backendReady = true;

      // React to every auth change (initial load, admin sign-in/out).
      auth.onAuthStateChanged(async (user) => {
        if (user) {
          await handleAuthedUser(user);
        } else {
          myUid = null;
          myHandle = null;
          adminMode = false;
          emitAuth();
          try {
            await auth.signInAnonymously();
          } catch (err) {
            lastAuthError = err.code || err.message || String(err);
            console.warn("Anonymous sign-in failed", err);
            markAuthReady();
          }
        }
      });

      // Kick off anonymous sign-in if nobody is signed in yet.
      if (!auth.currentUser) {
        try {
          await auth.signInAnonymously();
        } catch (err) {
          lastAuthError = err.code || err.message || String(err);
          console.warn("Anonymous sign-in failed", err);
          markAuthReady();
        }
      } else {
        await handleAuthedUser(auth.currentUser);
      }
    })();
    return readyPromise;
  }

  function parseDoc(doc) {
    const d = doc.data() || {};
    let created;
    if (d.createdAt && typeof d.createdAt.toDate === "function") {
      created = d.createdAt.toDate().toISOString();
    } else {
      created = new Date().toISOString();
    }
    return {
      id: doc.id,
      runId: d.runId || "",
      specId: d.specId || "",
      context: d.context || "spec",
      step: Number(d.step),
      uid: d.uid || "",
      handle: d.handle || "explorer_000",
      author: d.handle || "explorer_000",
      text: d.text || "",
      createdAt: created,
    };
  }

  async function listNotes() {
    await init();
    if (!backendReady) return [];
    const snap = await db
      .collection("notes")
      .orderBy("createdAt", "desc")
      .limit(1000)
      .get();
    return snap.docs.map(parseDoc).filter((n) => Number.isFinite(n.step));
  }

  function filterNotes(notes, { runId, specId, context }) {
    return notes.filter(
      (n) =>
        n.runId === runId &&
        n.specId === specId &&
        (n.context || "spec") === (context || "spec")
    );
  }

  async function createNote({ runId, specId, context, step, text }) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    await waitForUid();
    if (!myUid) throw new Error("Not signed in.");
    const clean = String(text || "").trim().slice(0, 2000);
    if (!clean) throw new Error("Note is empty.");
    let handle = adminMode ? "curator" : myHandle;
    if (!handle && !adminMode) {
      try {
        handle = await ensureHandle(myUid);
        myHandle = handle;
        emitAuth();
      } catch (err) {
        throw new Error(
          "Could not assign handle. Check Firestore rules were published. (" +
            (err.code || err.message || String(err)) +
            ")"
        );
      }
    }
    if (!handle) throw new Error("No handle assigned yet — try again in a moment.");
    const payload = {
      runId,
      specId,
      context: context || "spec",
      step: Math.round(step),
      uid: myUid,
      handle,
      text: clean,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    };
    const ref = await db.collection("notes").add(payload);
    return {
      mode: "api",
      note: {
        id: ref.id,
        runId,
        specId,
        context: context || "spec",
        step: Math.round(step),
        uid: myUid,
        handle,
        author: handle,
        text: clean,
        createdAt: new Date().toISOString(),
      },
    };
  }

  async function deleteNote(id) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete notes.");
    await db.collection("notes").doc(id).delete();
  }

  async function signInAdmin(email, password) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    const cred = await auth.signInWithEmailAndPassword(email, password);
    if (cred.user.uid !== adminUid()) {
      await auth.signOut();
      throw new Error("This account is not the curator.");
    }
    return true;
  }

  async function signOutAdmin() {
    await init();
    if (!backendReady) return;
    await auth.signOut(); // onAuthStateChanged re-signs in anonymously
  }

  function isAdmin() {
    return adminMode;
  }
  function currentHandle() {
    return adminMode ? "curator" : myHandle;
  }
  function isReady() {
    return backendReady;
  }

  return {
    init,
    onAuthChange,
    listNotes,
    filterNotes,
    createNote,
    deleteNote,
    signInAdmin,
    signOutAdmin,
    isAdmin,
    currentHandle,
    isReady,
    cfg,
  };
})();
