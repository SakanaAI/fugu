Codex version switches now save your session index

When Fugu switches your Codex version, it now copies your session index (the state, memories, and goals files under ~/.codex) into the same timestamped backup it already makes, and it warns you before the switch. Codex lists past sessions per version, so after a switch "codex resume" can show a different set of sessions.

Nothing is required on your part. Your session transcripts under ~/.codex/sessions are never deleted, and codex-fugu keeps working as before.

If a switch leaves "codex resume" showing fewer sessions than you expect, run the Codex version that wrote them to see them again, or restore the saved index from your latest backup:

  cp -p ~/.codex-backups/codex-config-<timestamp>/*.sqlite* ~/.codex/
