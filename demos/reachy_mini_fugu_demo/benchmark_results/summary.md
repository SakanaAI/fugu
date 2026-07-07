# Reachy Mini Simulation Comparison

Mode: `smoke`. Shared task: five typed user questions through the Reachy answer, schema, gesture, and TTS pipeline.

| model | run | valid JSON | schema valid | answer | gesture plan | unknown primitives | TTS | fallbacks | median model ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fugu_ultra | Fugu Ultra (xhigh) | 5/5 | 5/5 | 5/5 | 5/5 | none | 5/5 | 0/5 | 0 |
| fugu | Fugu (xhigh) | 5/5 | 5/5 | 5/5 | 5/5 | none | 5/5 | 0/5 | 0 |
| gpt | GPT-5.5 (xhigh) | 5/5 | 5/5 | 5/5 | 5/5 | none | 5/5 | 0/5 | 0 |
| gemini | Gemini 3.1 Pro (high) | 5/5 | 5/5 | 5/5 | 5/5 | none | 5/5 | 0/5 | 0 |
| opus | Claude Opus 4.8 (max) | 5/5 | 5/5 | 5/5 | 5/5 | none | 5/5 | 0/5 | 0 |
