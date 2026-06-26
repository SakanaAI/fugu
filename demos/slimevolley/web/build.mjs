// build.mjs — assemble the modular web/ source into one self-opening ../slimevolley.html.
// Inlines style.css, concatenates the JS modules (stripping ONLY their ES relative-imports/exports — never the
// Python `import` lines inside template strings), and bakes in the policy manifest + sources as window globals
// so the page opens straight from file:// with no server and no fetch. Run: node web/build.mjs

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));   // .../slimevolley/web
const demo = path.dirname(here);                             // .../slimevolley
const read = p => fs.readFileSync(p, 'utf8');

const ORDER = ['engine.js', 'rnn.js', 'pyrunner.js', 'manifest.js', 'match.js', 'render.js', 'panels.js', 'controls.js', 'app.js'];

// strip ONLY ES module syntax: relative `import ... from './x.js'` lines, and the `export ` keyword on
// declarations. (A Python `import sys` inside a template literal has no `from './...'` and no `export`, so it
// is left untouched.)
const strip = s => s
  .replace(/^[ \t]*import\b[^\n]*\bfrom\s+['"]\.\/[^\n]*\n/gm, '')
  .replace(/^([ \t]*)export\s+(?=(?:const|let|var|function|class|async)\b)/gm, '$1');

const js = ORDER.map(f => `// ===== ${f} =====\n${strip(read(path.join(here, f)))}`).join('\n');

// bake the curated manifest + each policy source in, so nothing is fetched at runtime
const manifest = JSON.parse(read(path.join(demo, 'policies.json')));
const srcs = {};
for (const p of manifest) if (p.file) srcs[p.id] = read(path.join(demo, p.file));
const data = `window.__POLICIES__ = ${JSON.stringify(manifest)};\nwindow.__POLICY_SRC__ = ${JSON.stringify(srcs)};\n`;

let html = read(path.join(here, 'index.html'));
html = html.replace('<link rel="stylesheet" href="style.css">', `<style>\n${read(path.join(here, 'style.css'))}\n</style>`);
html = html.replace('<script type="module" src="app.js"></script>', `<script>\n${data}\n${js}\n</script>`);

const out = path.join(demo, 'slimevolley.html');
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${(html.length / 1024).toFixed(1)} KB; ${manifest.length} policies)`);
