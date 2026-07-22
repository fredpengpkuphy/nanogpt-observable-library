# Anonymous notes — one-time setup (Firebase)

The chart notes are stored in **Firebase Firestore** instead of GitHub Issues.
This gives you exactly what you asked for:

- Anyone can post a note, fully **anonymous**.
- Every visitor keeps **one stable handle** (`explorer_001`, `explorer_002`, …)
  that stays the same across sessions and across every curve.
- **Only you** (the curator) can delete or edit notes.

The Firebase web config keys are **not secrets** — all access is enforced by the
security rules — so they are safe to commit to a public repo.

---

## 1. Create a Firebase project
1. Go to https://console.firebase.google.com → **Add project** (any name).
2. Google Analytics is optional; you can turn it off.

## 2. Enable Firestore
1. Left menu → **Build → Firestore Database → Create database**.
2. Choose **Production mode** (rules below lock it down), pick a region, done.

## 3. Enable authentication
1. **Build → Authentication → Get started**.
2. **Sign-in method** tab:
   - Enable **Anonymous**.
   - Enable **Email/Password**.
3. **Users** tab → **Add user** → enter *your* email + a password.
   - Click the new user → copy its **User UID**. This is your `adminUid`.

## 4. Copy the web config
1. **Project settings** (gear icon) → **General** → scroll to **Your apps**.
2. Click the **</>** (Web) icon, register an app (nickname only, no hosting).
3. Copy the `firebaseConfig` object values.
4. Paste them into `js/notes-config.js`:
   ```js
   firebase: {
     apiKey: "…",
     authDomain: "…",
     projectId: "…",
     storageBucket: "…",
     messagingSenderId: "…",
     appId: "…",
   },
   adminUid: "the UID you copied in step 3",
   ```

## 5. Publish the security rules
1. **Firestore Database → Rules**.
2. Paste the contents of `firestore.rules`.
3. Replace **`ADMIN_UID_HERE`** with the same UID from step 3.
4. **Publish**.

## 6. Authorize your domain
**Authentication → Settings → Authorized domains** → make sure your GitHub
Pages domain (e.g. `yourname.github.io`) and `localhost` are listed.

---

## Using it

- **Visitors**: open a chart fullscreen, click the curve at a step, write a note.
  They are auto-assigned `explorer_00N` on first post and keep it forever
  (unless they wipe browser storage).
- **You (curator)**: open the explorer with `?admin=1` in the URL, e.g.
  `explorer.html?run=…&admin=1`. A sign-in box appears in the Notes panel;
  sign in with your email/password. You'll then see a ✕ delete button on every
  note. Click **Sign out** to return to normal browsing.

## Notes on the design
- Handles are assigned by a Firestore transaction on `meta/counter`, so numbers
  are sequential and unique.
- A handle is tied to the browser's anonymous Firebase identity, which persists
  automatically. Clearing site data / using another browser yields a new number
  — unavoidable without forcing people to log in.
