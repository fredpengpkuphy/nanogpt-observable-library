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
- **You (curator)**: open any page with `?admin=1`, sign in with email/password.
  You'll see delete controls on notes. Click **Sign out** to leave curator mode.

## Using suggestions

- Public page: `suggestions.html` — anyone can post improvement ideas (anonymous).
- Only the curator can reply; replies are public. Curator can also delete.
- After updating `firestore.rules`, **Publish** again in Firebase (adds `suggestions`).

## Using announcements

- Public list: `announcements.html` (also linked from Select / Formulas / Explorer).
- Banner on every page shows the newest 3; **All →** opens the full list.
- Curator: open **Announcements →** from the curator bar, then Publish / Edit / Delete.
- After updating `firestore.rules`, **Publish** the rules in the Firebase console
  (needed for the new `announcements` collection).
- If you still have the old single `meta/siteAnnouncement`, it appears as
  **legacy** on the announcements page — click **Move into list** once.

## Notes on the design
- Notes are anonymous (no public usernames).
- A visitor's Firebase anonymous identity persists in the browser; clearing site
  data yields a new identity — unavoidable without forcing people to log in.
