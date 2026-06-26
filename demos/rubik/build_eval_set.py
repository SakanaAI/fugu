"""
Freeze the evaluation set for the Rubik model comparison.

Generates a deterministic set of scrambled cubes and writes them to
eval_cubes.json. Cube #0 (seed 2026) is the "hero" cube used for the visual /
story; the rest form the held-out set for a robust mean-turns + solve-rate
number. The scramble move lists are recorded for the record but are NOT given
to the models (only the facelet strings are).

Run:  python3 build_eval_set.py
"""
import json
from pathlib import Path

from cube import scramble, count_turns, from_facelet, is_solved

HERO_SEED = 2026
N_CUBES = 300         # hero + 299 held-out (first 100 seeds unchanged from the 100-cube set)
SCRAMBLE_LEN = 25     # HTM random moves per scramble (no same-face repeats)
OUT = Path(__file__).with_name("eval_cubes.json")


def main():
    seeds = [HERO_SEED] + [HERO_SEED + i for i in range(1, N_CUBES)]
    cubes = []
    for idx, seed in enumerate(seeds):
        facelet, moves = scramble(n=SCRAMBLE_LEN, seed=seed)
        # sanity: the recorded scramble really produces this facelet & is not solved
        assert not is_solved(from_facelet(facelet)), f"cube {idx} came out solved"
        cubes.append({
            "id": idx,
            "hero": idx == 0,
            "seed": seed,
            "scramble_moves": " ".join(moves),  
            "scramble_len": count_turns(moves),
            "facelet": facelet,                   # the ONLY thing given to models
        })
    OUT.write_text(json.dumps(cubes, indent=2), encoding="utf-8")
    print(f"wrote {len(cubes)} cubes -> {OUT}")
    print(f"hero (id 0, seed {HERO_SEED}): {cubes[0]['facelet']}")


if __name__ == "__main__":
    main()
