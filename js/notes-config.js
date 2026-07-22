/**
 * Anonymous chart notes backed by Firebase (Firestore + Auth).
 *
 * Anyone can post notes / comments anonymously (no public usernames).
 * Only the curator (email/password) can delete notes or comments.
 * Sign in via ?admin=1 on the homepage or explorer.
 *
 * Setup: see NOTES_SETUP.md and firestore.rules (must Publish rules after changes).
 */
window.NOTES_CONFIG = {
  firebase: {
    apiKey: "AIzaSyC4Vf_pleoXbFfNcXxf0LxnFiYI9UFZJ54",
    authDomain: "nanogpt-observable-explorer.firebaseapp.com",
    projectId: "nanogpt-observable-explorer",
    storageBucket: "nanogpt-observable-explorer.firebasestorage.app",
    messagingSenderId: "947844170617",
    appId: "1:947844170617:web:54c6084ccf263e3a5865a8",
  },

  // Curator UID (Authentication → Users). Only this account may delete.
  adminUid: "XnMi4PnWRIV722yxz3khH7uFHOx1",
};
