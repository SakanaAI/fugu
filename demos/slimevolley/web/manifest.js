// manifest.js: load the curated policy list + each policy's source. Dual-path so the SAME code runs in the
// built single file (everything inlined as window.__POLICIES__ / __POLICY_SRC__) and in the modular dev page
// (served, fetched from the parent dir). '../' reaches the demo root from web/ regardless of the server root.

export async function loadManifest() {
  if (typeof window !== 'undefined' && window.__POLICIES__) return window.__POLICIES__;
  return (await fetch('../policies.json')).json();
}

export async function loadPolicySource(p) {
  if (typeof window !== 'undefined' && window.__POLICY_SRC__ && window.__POLICY_SRC__[p.id]) {
    return window.__POLICY_SRC__[p.id];
  }
  return (await fetch('../' + p.file)).text();
}

// Optional, HTTP-only: a git-ignored train/run/inventory.json mapping {run-name: champion-path} that the
// trainer writes (paths relative to that file's own dir). Lets a served demo show retrained champions next
// to the shipped policies. Returns [] when absent (file://, or not served), so the shipped page is
// unaffected. The built page sits at the demo root, the dev page under web/, so the base path differs.
// Each entry loads independently: a stale or missing champion is skipped, never sinking the whole list.
export async function loadExtraInventory() {
  const base = (typeof window !== 'undefined' && window.__POLICIES__) ? 'train/run/' : '../train/run/';
  let inventory;
  try {
    const res = await fetch(base + 'inventory.json', { cache: 'no-store' });
    if (!res.ok) return [];
    inventory = await res.json();
  } catch {
    return [];   // no inventory served (file://, or not served): the shipped policies only
  }
  const entries = [];
  for (const [name, path] of Object.entries(inventory)) {
    try {
      const r = await fetch(base + path);
      if (!r.ok) continue;   // a stale or missing champion file: skip just this entry, keep the rest
      entries.push({ id: name + '-retrain', label: name + ' (retrained)', kind: 'code', source: await r.text() });
    } catch {
      // network error on this entry: skip it, keep the rest
    }
  }
  return entries;
}
