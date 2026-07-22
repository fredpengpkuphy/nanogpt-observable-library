/**
 * Anonymous, persistent chart notes backed by Firebase (Firestore + Auth).
 *
 * Why Firebase?  The site is static (GitHub Pages, no server), yet we need:
 *   - anyone can post anonymously,
 *   - only the site owner can delete/manage,
 *   - every visitor keeps ONE stable handle (explorer_001) across sessions
 *     and across every curve.
 * Firestore security rules enforce all of this without any backend code.
 *
 * ── One-time setup (see NOTES_SETUP.md for screenshots) ─────────────────────
 * 1. Create a free Firebase project → https://console.firebase.google.com
 * 2. Build → Firestore Database → Create database (production mode).
 * 3. Build → Authentication → Sign-in method:
 *      - enable "Anonymous"
 *      - enable "Email/Password"
 *      Then Authentication → Users → "Add user" with YOUR email + password.
 *      Copy that user's UID into `adminUid` below.
 * 4. Project settings → General → "Your apps" → Web app → copy the config
 *    object into `firebase` below.
 * 5. Firestore → Rules → paste the contents of firestore.rules
 *    (replace ADMIN_UID_HERE with the same UID) → Publish.
 *
 * The Firebase web config keys are NOT secrets — access is controlled by the
 * security rules, so it is safe to commit them to a public repo.
 */
window.NOTES_CONFIG = {
  // Paste from Firebase console → Project settings → Your apps → Web app.
  firebase: {
    apiKey: "AIzaSyC4Vf_pleoXbFfNcXxf0LxnFiYI9UFZJ54",
    authDomain: "nanogpt-observable-explorer.firebaseapp.com",
    projectId: "nanogpt-observable-explorer",
    storageBucket: "nanogpt-observable-explorer.firebasestorage.app",
    messagingSenderId: "947844170617",
    appId: "1:947844170617:web:54c6084ccf263e3a5865a8",
  },

  // UID of the admin user (the email/password account you created in step 3).
  // Only this account may delete or edit notes. Sign in via ?admin=1.
  adminUid: "XnMi4PnWRIV722yxz3khH7uFHOx1",

  // Format for auto-assigned anonymous handles: explorer_001, explorer_002, …
  handlePrefix: "explorer_",
  handlePad: 3,
};
