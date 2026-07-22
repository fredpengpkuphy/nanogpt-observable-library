/**
 * Chart notes backed by Firebase (Firestore + Auth).
 *
 * Collections
 *   notes/{id}
 *     { runId, specId, context, step|null, uid, text, createdAt }
 *   notes/{id}/comments/{id}
 *     { uid, text, parentId|null, createdAt }
 *
 * Visitors sign in anonymously under the hood (no public username).
 * Only the curator (email/password) may delete notes or comments.
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

  function emitAuth() {
    const state = {
      ready: backendReady,
      uid: myUid,
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
    fn({ ready: backendReady, uid: myUid, isAdmin: adminMode });
    return () => authListeners.delete(fn);
  }

  async function handleAuthedUser(user) {
    myUid = user.uid;
    lastAuthError = null;
    adminMode = user.uid === adminUid() && !user.isAnonymous;
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

      auth.onAuthStateChanged(async (user) => {
        if (user) {
          await handleAuthedUser(user);
        } else {
          myUid = null;
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

  function tsToIso(value) {
    if (value && typeof value.toDate === "function") {
      return value.toDate().toISOString();
    }
    return new Date().toISOString();
  }

  function parseNote(doc) {
    const d = doc.data() || {};
    const stepRaw = d.step;
    const step =
      stepRaw === null || stepRaw === undefined || stepRaw === ""
        ? null
        : Number(stepRaw);
    return {
      id: doc.id,
      runId: d.runId || "",
      specId: d.specId || "",
      context: d.context || "spec",
      step: Number.isFinite(step) ? step : null,
      uid: d.uid || "",
      text: d.text || "",
      createdAt: tsToIso(d.createdAt),
      comments: [],
    };
  }

  function parseComment(doc) {
    const d = doc.data() || {};
    return {
      id: doc.id,
      noteId: d.noteId || "",
      uid: d.uid || "",
      text: d.text || "",
      parentId: d.parentId || null,
      createdAt: tsToIso(d.createdAt),
    };
  }

  async function listComments(noteId) {
    const snap = await db
      .collection("notes")
      .doc(noteId)
      .collection("comments")
      .orderBy("createdAt", "asc")
      .limit(200)
      .get();
    return snap.docs.map(parseComment);
  }

  async function listNotes() {
    await init();
    if (!backendReady) return [];
    const snap = await db
      .collection("notes")
      .orderBy("createdAt", "desc")
      .limit(500)
      .get();
    const notes = snap.docs.map(parseNote);
    await Promise.all(
      notes.map(async (n) => {
        try {
          n.comments = await listComments(n.id);
        } catch (err) {
          console.warn("listComments failed", n.id, err);
          n.comments = [];
        }
      })
    );
    return notes;
  }

  function filterNotes(notes, { runId, specId, context }) {
    return notes.filter(
      (n) =>
        n.runId === runId &&
        n.specId === specId &&
        (n.context || "spec") === (context || "spec")
    );
  }

  function normalizeStep(step) {
    if (step === null || step === undefined || step === "") return null;
    const n = Number(step);
    return Number.isFinite(n) ? Math.round(n) : null;
  }

  async function createNote({ runId, specId, context, step = null, text }) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    await waitForUid();
    if (!myUid) throw new Error("Not signed in.");
    const clean = String(text || "").trim().slice(0, 2000);
    if (!clean) throw new Error("Note is empty.");
    const stepVal = normalizeStep(step);
    const payload = {
      runId,
      specId,
      context: context || "spec",
      step: stepVal,
      uid: myUid,
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
        step: stepVal,
        uid: myUid,
        text: clean,
        createdAt: new Date().toISOString(),
        comments: [],
      },
    };
  }

  async function createComment({ noteId, text, parentId = null }) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    await waitForUid();
    if (!myUid) throw new Error("Not signed in.");
    const clean = String(text || "").trim().slice(0, 2000);
    if (!clean) throw new Error("Comment is empty.");
    if (!noteId) throw new Error("Missing note id.");
    const payload = {
      noteId,
      uid: myUid,
      text: clean,
      parentId: parentId || null,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    };
    const ref = await db
      .collection("notes")
      .doc(noteId)
      .collection("comments")
      .add(payload);
    return {
      id: ref.id,
      noteId,
      uid: myUid,
      text: clean,
      parentId: parentId || null,
      createdAt: new Date().toISOString(),
    };
  }

  async function deleteNote(id) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete notes.");
    const noteRef = db.collection("notes").doc(id);
    const comments = await noteRef.collection("comments").limit(400).get();
    const batch = db.batch();
    comments.docs.forEach((d) => batch.delete(d.ref));
    batch.delete(noteRef);
    await batch.commit();
  }

  async function deleteComment(noteId, commentId) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete comments.");
    const col = db.collection("notes").doc(noteId).collection("comments");
    const snap = await col.limit(400).get();
    const all = snap.docs.map((d) => ({ id: d.id, ...d.data(), ref: d.ref }));
    const toDelete = new Set([commentId]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const c of all) {
        if (c.parentId && toDelete.has(c.parentId) && !toDelete.has(c.id)) {
          toDelete.add(c.id);
          grew = true;
        }
      }
    }
    const batch = db.batch();
    for (const c of all) {
      if (toDelete.has(c.id)) batch.delete(c.ref);
    }
    await batch.commit();
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
    await auth.signOut();
  }

  function isAdmin() {
    return adminMode;
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
    createComment,
    deleteNote,
    deleteComment,
    signInAdmin,
    signOutAdmin,
    isAdmin,
    isReady,
    cfg,
  };
})();
