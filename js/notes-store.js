/**
 * Chart notes backed by Firebase (Firestore + Auth).
 *
 * Collections
 *   notes/{id}
 *     { runId, specId, context, step|null, uid, text, createdAt }
 *   notes/{id}/comments/{id}
 *     { uid, text, parentId|null, createdAt }
 *   announcements/{id}
 *     { text, uid, createdAt, updatedAt }
 *   suggestions/{id}
 *     { text, uid, createdAt }
 *   suggestions/{id}/replies/{id}
 *     { text, uid, createdAt }  // curator-only
 *
 * Visitors sign in anonymously under the hood (no public username).
 * Only the curator (email/password) may delete notes/comments, manage announcements,
 * or reply to site suggestions.
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
  const BATCH_WRITE_LIMIT = 450;
  const READ_CONCURRENCY = 8;

  async function mapConcurrent(items, limit, mapper) {
    const results = new Array(items.length);
    let next = 0;
    async function worker() {
      while (true) {
        const index = next;
        next += 1;
        if (index >= items.length) return;
        results[index] = await mapper(items[index], index);
      }
    }
    const workerCount = Math.min(Math.max(1, limit), items.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
    return results;
  }

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
    if (typeof value === "string" || typeof value === "number") {
      const date = new Date(value);
      if (!Number.isNaN(date.getTime())) return date.toISOString();
    }
    return "";
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

  async function listNotes(filter = null) {
    await init();
    if (!backendReady) return [];
    let query = db.collection("notes");
    if (filter?.runId) query = query.where("runId", "==", filter.runId);
    if (filter?.specId) query = query.where("specId", "==", filter.specId);
    if (filter?.context) query = query.where("context", "==", filter.context);
    let snap;
    try {
      snap = await query.limit(500).get();
    } catch (err) {
      // Older projects may need a composite index for the precise query.
      // A run-only query remains correct after the client-side filter.
      if (!filter?.runId) throw err;
      snap = await db.collection("notes").where("runId", "==", filter.runId).limit(1000).get();
    }
    let notes = snap.docs.map(parseNote);
    if (filter) notes = filterNotes(notes, filter);
    notes.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
    await mapConcurrent(
      notes,
      READ_CONCURRENCY,
      async (n) => {
        try {
          n.comments = await listComments(n.id);
        } catch (err) {
          console.warn("listComments failed", n.id, err);
          n.comments = [];
        }
      }
    );
    return notes;
  }

  async function deleteCollection(collectionRef) {
    while (true) {
      const snap = await collectionRef.limit(BATCH_WRITE_LIMIT).get();
      if (snap.empty) return;
      const batch = db.batch();
      snap.docs.forEach((doc) => batch.delete(doc.ref));
      await batch.commit();
      if (snap.size < BATCH_WRITE_LIMIT) return;
    }
  }

  async function readCollection(collectionRef) {
    const docs = [];
    let last = null;
    while (true) {
      let query = collectionRef
        .orderBy(firebase.firestore.FieldPath.documentId())
        .limit(BATCH_WRITE_LIMIT);
      if (last) query = query.startAfter(last);
      const snap = await query.get();
      docs.push(...snap.docs);
      if (snap.size < BATCH_WRITE_LIMIT) return docs;
      last = snap.docs[snap.docs.length - 1];
    }
  }

  async function deleteRefs(refs) {
    for (let i = 0; i < refs.length; i += BATCH_WRITE_LIMIT) {
      const batch = db.batch();
      refs.slice(i, i + BATCH_WRITE_LIMIT).forEach((ref) => batch.delete(ref));
      await batch.commit();
    }
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
    await deleteCollection(noteRef.collection("comments"));
    await noteRef.delete();
  }

  async function deleteComment(noteId, commentId) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete comments.");
    const col = db.collection("notes").doc(noteId).collection("comments");
    const docs = await readCollection(col);
    const all = docs.map((d) => ({ id: d.id, ...d.data(), ref: d.ref }));
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
    await deleteRefs(all.filter((c) => toDelete.has(c.id)).map((c) => c.ref));
  }

  function friendlyAdminSignInError(err) {
    const code = String(err && err.code ? err.code : "");
    if (
      code === "auth/invalid-credential" ||
      code === "auth/wrong-password" ||
      code === "auth/user-not-found" ||
      code === "auth/invalid-email" ||
      code === "auth/invalid-login-credentials"
    ) {
      return "Wrong email or password.";
    }
    if (code === "auth/too-many-requests") {
      return "Too many attempts. Try again later.";
    }
    if (code === "auth/network-request-failed") {
      return "Network error. Check your connection.";
    }
    if (code === "auth/user-disabled") {
      return "This account is disabled.";
    }
    if (err && err.message === "This account is not the curator.") {
      return err.message;
    }
    if (err && err.message === "Notes backend is not configured yet.") {
      return err.message;
    }
    return "Sign-in failed. Please try again.";
  }

  async function signInAdmin(email, password) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    try {
      const cred = await auth.signInWithEmailAndPassword(email, password);
      if (cred.user.uid !== adminUid()) {
        await auth.signOut();
        throw new Error("This account is not the curator.");
      }
      return true;
    } catch (err) {
      throw new Error(friendlyAdminSignInError(err));
    }
  }

  async function signOutAdmin() {
    await init();
    if (!backendReady) return;
    await auth.signOut();
  }

  function mapAnnouncementDoc(doc) {
    const d = doc.data() || {};
    return {
      id: doc.id,
      text: String(d.text || "").trim(),
      createdAt: d.createdAt || null,
      updatedAt: d.updatedAt || null,
      legacy: false,
    };
  }

  async function readLegacyAnnouncement() {
    const snap = await db.collection("meta").doc("siteAnnouncement").get();
    if (!snap.exists) return null;
    const text = String(snap.data().text || "").trim();
    if (!text) return null;
    return {
      id: "_legacy",
      text,
      createdAt: snap.data().updatedAt || null,
      updatedAt: snap.data().updatedAt || null,
      legacy: true,
    };
  }

  async function listAnnouncements() {
    await init();
    if (!backendReady) return [];
    const snap = await db
      .collection("announcements")
      .orderBy("createdAt", "desc")
      .get();
    const items = snap.docs.map(mapAnnouncementDoc).filter((a) => a.text);
    if (items.length) return items;
    const legacy = await readLegacyAnnouncement();
    return legacy ? [legacy] : [];
  }

  /** Live list for banner + announcements page. Callback receives an array. */
  function watchAnnouncements(cb) {
    let unsub = null;
    let cancelled = false;
    let generation = 0;
    init().then(async () => {
      if (cancelled) return;
      if (!backendReady) {
        cb([]);
        return;
      }
      unsub = db
        .collection("announcements")
        .orderBy("createdAt", "desc")
        .onSnapshot(
          async (snap) => {
            const currentGeneration = ++generation;
            const items = snap.docs.map(mapAnnouncementDoc).filter((a) => a.text);
            if (items.length) {
              if (!cancelled && currentGeneration === generation) cb(items);
              return;
            }
            try {
              const legacy = await readLegacyAnnouncement();
              if (!cancelled && currentGeneration === generation) {
                cb(legacy ? [legacy] : []);
              }
            } catch (_) {
              if (!cancelled && currentGeneration === generation) cb([]);
            }
          },
          (err) => {
            console.warn("watchAnnouncements", err);
            cb([]);
          }
        );
    });
    return () => {
      cancelled = true;
      if (unsub) unsub();
    };
  }

  async function createAnnouncement(text) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can publish announcements.");
    const clean = String(text || "").trim().slice(0, 1000);
    if (!clean) throw new Error("Announcement text is empty.");
    const ref = await db.collection("announcements").add({
      text: clean,
      uid: myUid,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
      updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
    });
    return { id: ref.id, text: clean, legacy: false };
  }

  async function updateAnnouncement(id, text) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can edit announcements.");
    const clean = String(text || "").trim().slice(0, 1000);
    if (!clean) throw new Error("Announcement text is empty.");
    if (id === "_legacy") {
      await db.collection("meta").doc("siteAnnouncement").set({
        text: clean,
        uid: myUid,
        updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
      });
      return { id: "_legacy", text: clean, legacy: true };
    }
    await db.collection("announcements").doc(id).update({
      text: clean,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
    });
    return { id, text: clean, legacy: false };
  }

  async function deleteAnnouncement(id) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete announcements.");
    if (id === "_legacy") {
      await db.collection("meta").doc("siteAnnouncement").delete();
      return;
    }
    await db.collection("announcements").doc(id).delete();
  }

  /** Migrate the old single meta/siteAnnouncement into the collection. */
  async function migrateLegacyAnnouncement() {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can migrate announcements.");
    const legacy = await readLegacyAnnouncement();
    if (!legacy) return null;
    const created = await createAnnouncement(legacy.text);
    await db.collection("meta").doc("siteAnnouncement").delete();
    return created;
  }

  /** @deprecated Prefer listAnnouncements / createAnnouncement. */
  async function getAnnouncement() {
    const items = await listAnnouncements();
    return items[0] ? items[0].text : "";
  }

  /** @deprecated Prefer watchAnnouncements. */
  function watchAnnouncement(cb) {
    return watchAnnouncements((items) => {
      cb(items[0] ? items[0].text : "");
    });
  }

  /** @deprecated Prefer createAnnouncement / deleteAnnouncement. */
  async function setAnnouncement(text) {
    const clean = String(text || "").trim();
    if (!clean) {
      const items = await listAnnouncements();
      for (const item of items) await deleteAnnouncement(item.id);
      return "";
    }
    await createAnnouncement(clean);
    return clean;
  }

  async function loadSuggestionReplies(docRef) {
    const snap = await docRef.collection("replies").orderBy("createdAt", "asc").get();
    return snap.docs.map((d) => {
      const data = d.data() || {};
      return {
        id: d.id,
        text: String(data.text || "").trim(),
        createdAt: data.createdAt || null,
      };
    });
  }

  async function listSuggestions() {
    await init();
    if (!backendReady) return [];
    const snap = await db.collection("suggestions").orderBy("createdAt", "desc").get();
    const items = await mapConcurrent(snap.docs, READ_CONCURRENCY, async (doc) => {
      const data = doc.data() || {};
      const text = String(data.text || "").trim();
      if (!text) return null;
      return {
        id: doc.id,
        text,
        createdAt: data.createdAt || null,
        replies: await loadSuggestionReplies(doc.ref),
      };
    });
    return items.filter(Boolean);
  }

  function watchSuggestions(cb) {
    let unsub = null;
    let cancelled = false;
    let generation = 0;
    init().then(() => {
      if (cancelled) return;
      if (!backendReady) {
        cb([]);
        return;
      }
      unsub = db
        .collection("suggestions")
        .orderBy("createdAt", "desc")
        .onSnapshot(
          async (snap) => {
            const currentGeneration = ++generation;
            try {
              const items = await mapConcurrent(
                snap.docs,
                READ_CONCURRENCY,
                async (doc) => {
                  const data = doc.data() || {};
                  const text = String(data.text || "").trim();
                  if (!text) return null;
                  return {
                    id: doc.id,
                    text,
                    createdAt: data.createdAt || null,
                    replies: await loadSuggestionReplies(doc.ref),
                  };
                }
              );
              if (!cancelled && currentGeneration === generation) {
                cb(items.filter(Boolean));
              }
            } catch (err) {
              console.warn("watchSuggestions replies", err);
              if (!cancelled && currentGeneration === generation) cb([]);
            }
          },
          (err) => {
            console.warn("watchSuggestions", err);
            if (!cancelled) cb([]);
          }
        );
    });
    return () => {
      cancelled = true;
      if (unsub) unsub();
    };
  }

  async function createSuggestion(text) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    await waitForUid();
    if (!myUid) throw new Error("Not signed in.");
    const clean = String(text || "").trim().slice(0, 2000);
    if (!clean) throw new Error("Suggestion is empty.");
    const ref = await db.collection("suggestions").add({
      text: clean,
      uid: myUid,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    });
    return { id: ref.id, text: clean, replies: [] };
  }

  async function replyToSuggestion(suggestionId, text) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can reply to suggestions.");
    const clean = String(text || "").trim().slice(0, 2000);
    if (!clean) throw new Error("Reply is empty.");
    const parent = db.collection("suggestions").doc(suggestionId);
    const parentSnap = await parent.get();
    if (!parentSnap.exists) throw new Error("Suggestion not found.");
    const ref = await parent.collection("replies").add({
      text: clean,
      uid: myUid,
      createdAt: firebase.firestore.FieldValue.serverTimestamp(),
    });
    return { id: ref.id, text: clean };
  }

  async function deleteSuggestion(id) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete suggestions.");
    const parent = db.collection("suggestions").doc(id);
    await deleteCollection(parent.collection("replies"));
    await parent.delete();
  }

  async function deleteSuggestionReply(suggestionId, replyId) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can delete replies.");
    await db
      .collection("suggestions")
      .doc(suggestionId)
      .collection("replies")
      .doc(replyId)
      .delete();
  }

  async function getMaintenanceMode() {
    await init();
    if (!backendReady) return { enabled: false };
    const snap = await db.collection("meta").doc("siteMaintenance").get();
    if (!snap.exists) return { enabled: false };
    return { enabled: !!snap.data().enabled };
  }

  function watchMaintenanceMode(cb) {
    let unsub = null;
    let cancelled = false;
    init().then(() => {
      if (cancelled) return;
      if (!backendReady) {
        cb({ enabled: false });
        return;
      }
      unsub = db.collection("meta").doc("siteMaintenance").onSnapshot(
        (snap) => {
          cb({ enabled: snap.exists ? !!snap.data().enabled : false });
        },
        (err) => {
          console.warn("watchMaintenanceMode", err);
          cb({ enabled: false });
        }
      );
    });
    return () => {
      cancelled = true;
      if (unsub) unsub();
    };
  }

  async function setMaintenanceMode(enabled) {
    await init();
    if (!backendReady) throw new Error("Notes backend is not configured yet.");
    if (!adminMode) throw new Error("Only the curator can change maintenance mode.");
    const on = !!enabled;
    await db.collection("meta").doc("siteMaintenance").set({
      enabled: on,
      uid: myUid,
      updatedAt: firebase.firestore.FieldValue.serverTimestamp(),
    });
    return on;
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
    listAnnouncements,
    watchAnnouncements,
    createAnnouncement,
    updateAnnouncement,
    deleteAnnouncement,
    migrateLegacyAnnouncement,
    getAnnouncement,
    watchAnnouncement,
    setAnnouncement,
    listSuggestions,
    watchSuggestions,
    createSuggestion,
    replyToSuggestion,
    deleteSuggestion,
    deleteSuggestionReply,
    getMaintenanceMode,
    watchMaintenanceMode,
    setMaintenanceMode,
    signInAdmin,
    signOutAdmin,
    isAdmin,
    isReady,
    cfg,
  };
})();
