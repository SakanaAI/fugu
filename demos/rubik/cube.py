"""
Self-contained Rubik's cube engine + verifier for the Fugu model comparison.

This is OUR tooling (scramble generator + ground-truth verifier). The models
under test must write their OWN solver from scratch; this file is never given
to them. Pure standard library, no dependencies, so the verifier is trivially
trustworthy and reproducible.

FACELET STRING CONVENTION  (the standard "Kociemba / Singmaster" layout)
  54 chars, faces concatenated in the order   U R F D L B.
  Each face is 9 facelets read row-major (top-left -> bottom-right) in the
  standard unfolded-net orientation. Each char names the face whose colour the
  facelet shows in the solved cube (one of U R F D L B).

  index ranges:  U 0..8   R 9..17   F 18..26   D 27..35   L 36..44   B 45..53

  solved cube:
    UUUUUUUUU RRRRRRRRR FFFFFFFFF DDDDDDDDD LLLLLLLLL BBBBBBBBB

MOVE NOTATION  (standard WCA)
  U D L R F B   = 90 degrees CLOCKWISE looking at that face from OUTSIDE.
  X'  = counter-clockwise  (= 3 x clockwise).      X2 = 180 degrees.
TURN METRIC: HTM (half-turn metric) -- each of X, X', X2 counts as ONE turn.

The engine is GEOMETRIC: it rotates the actual 3D stickers, so move semantics
come from real rotation matrices, not hand-typed permutation tables. The only
convention-sensitive part is the facelet layout, which is pinned by the
absolute-orientation anchor tests in selftest().

Coordinate frame:  x = right(+)/left(-),  y = up(+)/down(-),  z = front(+)/back(-).
"""
import random

SOLVED = "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9

# face -> (axis index, layer value along that axis, sign for a CLOCKWISE turn)
# positive-axis faces (U,R,F) turn one way, negative-axis faces (D,L,B) the other.
_FACES = {
    "U": (1, 1, -1),
    "D": (1, -1, 1),
    "R": (0, 1, -1),
    "L": (0, -1, 1),
    "F": (2, 1, -1),
    "B": (2, -1, 1),
}
_SUFFIX_REPS = {"": 1, "'": 3, "2": 2}
_INV_SUFFIX = {"": "'", "'": "", "2": "2"}
ALL_MOVES = [f + s for f in "UDLRFB" for s in ("", "'", "2")]  # 18 HTM moves


def _rot(coord, axis, s):
    """Rotate an integer 3-vector 90 degrees about a principal axis.
    s = +1 or -1 selects direction (right-handed +90 vs -90)."""
    x, y, z = coord
    if axis == 1:                      # about y:  +1 -> (x,z)->(z,-x)
        return (z, y, -x) if s > 0 else (-z, y, x)
    if axis == 0:                      # about x:  +1 -> (y,z)->(-z,y)
        return (x, -z, y) if s > 0 else (x, z, -y)
    return (-y, x, z) if s > 0 else (y, -x, z)   # about z: +1 -> (x,y)->(-y,x)


def _slots():
    """Ordered list of (face_letter, pos, normal) for all 54 facelets, in the
    U,R,F,D,L,B facelet-string order. pos/normal are int 3-tuples in {-1,0,1}."""
    out = []

    def add(face, normal, coordfn):
        for row in range(3):
            for col in range(3):
                out.append((face, coordfn(row, col), normal))

    add("U", (0, 1, 0),  lambda r, c: (c - 1, 1, r - 1))
    add("R", (1, 0, 0),  lambda r, c: (1, 1 - r, 1 - c))
    add("F", (0, 0, 1),  lambda r, c: (c - 1, 1 - r, 1))
    add("D", (0, -1, 0), lambda r, c: (c - 1, -1, 1 - r))
    add("L", (-1, 0, 0), lambda r, c: (-1, 1 - r, c - 1))
    add("B", (0, 0, -1), lambda r, c: (1 - c, 1 - r, -1))
    return out


_SLOTS = _slots()


def solved_state():
    """State = dict mapping (pos, normal) -> colour letter."""
    return {(pos, nrm): face for (face, pos, nrm) in _SLOTS}


def to_facelet(state):
    return "".join(state[(pos, nrm)] for (_f, pos, nrm) in _SLOTS)


def from_facelet(s):
    s = s.strip()
    if len(s) != 54:
        raise ValueError(f"facelet string must be 54 chars, got {len(s)}")
    return {(pos, nrm): s[i] for i, (_f, pos, nrm) in enumerate(_SLOTS)}


def is_solved(state):
    return to_facelet(state) == SOLVED


def _turn_cw(state, face):
    axis, lval, s = _FACES[face]
    new = {}
    for (pos, nrm), color in state.items():
        if pos[axis] == lval:
            new[(_rot(pos, axis, s), _rot(nrm, axis, s))] = color
        else:
            new[(pos, nrm)] = color
    return new


def parse_moves(seq):
    """Accept a string ('R U2 F') or an iterable of tokens. Validate strictly."""
    toks = seq.split() if isinstance(seq, str) else list(seq)
    out = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if t[0] not in "UDLRFB" or t[1:] not in _SUFFIX_REPS:
            raise ValueError(f"bad move token: {t!r}")
        out.append(t)
    return out


def apply_moves(state, seq):
    for t in parse_moves(seq):
        face, suffix = t[0], t[1:]
        for _ in range(_SUFFIX_REPS[suffix]):
            state = _turn_cw(state, face)
    return state


def invert_moves(seq):
    toks = parse_moves(seq)
    return [t[0] + _INV_SUFFIX[t[1:]] for t in reversed(toks)]


def count_turns(seq):
    """HTM turn count: every token (X, X', X2) counts as one."""
    return len(parse_moves(seq))


def scramble(n=25, seed=0):
    """Apply n random HTM moves to a solved cube (no same-face repeats).
    Returns (facelet_string, scramble_move_list). The move list is kept private
    -- only the facelet string is given to the models."""
    rng = random.Random(seed)
    moves, prev = [], None
    for _ in range(n):
        m = rng.choice(ALL_MOVES)
        while m[0] == prev:
            m = rng.choice(ALL_MOVES)
        prev = m[0]
        moves.append(m)
    return to_facelet(apply_moves(solved_state(), moves)), moves


def verify_solution(scramble_facelet, solution_seq):
    """Apply a candidate solution to a scrambled state; return (solved?, n_turns).
    Raises ValueError on malformed move tokens."""
    state = from_facelet(scramble_facelet)
    state = apply_moves(state, solution_seq)
    return is_solved(state), count_turns(solution_seq)


# ---------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    s0 = solved_state()
    check("solved string is canonical", to_facelet(s0) == SOLVED)
    check("is_solved(solved)", is_solved(s0))
    check("facelet round-trip", to_facelet(from_facelet(SOLVED)) == SOLVED)

    # Each face^4 = identity.
    for f in "UDLRFB":
        st = s0
        for _ in range(4):
            st = _turn_cw(st, f)
        check(f"{f}^4 == identity", is_solved(st))

    # Each token, then its inverse, = identity; and X2 == X X.
    for f in "UDLRFB":
        for suf in ("", "'", "2"):
            tok = f + suf
            st = apply_moves(s0, [tok])
            st = apply_moves(st, invert_moves([tok]))
            check(f"{tok} then inverse == identity", is_solved(st))
        st2 = apply_moves(s0, [f + "2"])
        stxx = apply_moves(s0, [f, f])
        check(f"{f}2 == {f} {f}", to_facelet(st2) == to_facelet(stxx))

    # ABSOLUTE-ORIENTATION ANCHORS -- pin our notation to the world standard.
    # After one clockwise U, the strip that arrives at each face's TOP row is
    # the next face clockwise-from-above:  F-top<-R, R-top<-B, B-top<-L, L-top<-F.
    fs = to_facelet(apply_moves(s0, ["U"]))
    check("U anchor: F-top = RRR", fs[18:21] == "RRR")
    check("U anchor: R-top = BBB", fs[9:12] == "BBB")
    check("U anchor: B-top = LLL", fs[45:48] == "LLL")
    check("U anchor: L-top = FFF", fs[36:39] == "FFF")
    check("U anchor: U centre unchanged", fs[4] == "U")

    # After one clockwise R, the right column lifts front->up->back->down->front.
    fr = to_facelet(apply_moves(s0, ["R"]))
    check("R anchor: U right col = FFF", fr[2] + fr[5] + fr[8] == "FFF")
    check("R anchor: F right col = DDD", fr[20] + fr[23] + fr[26] == "DDD")
    check("R anchor: D right col = BBB", fr[29] + fr[32] + fr[35] == "BBB")

    # Non-trivial identity exercising two faces together.
    st = s0
    for _ in range(6):
        st = apply_moves(st, ["R", "U", "R'", "U'"])
    check("(R U R' U')^6 == identity", is_solved(st))

    # Scramble then its exact inverse returns to solved (random fuzz).
    rt_ok = True
    for seed in range(50):
        fac, mv = scramble(n=25, seed=seed)
        solved_back, _ = verify_solution(fac, invert_moves(mv))
        rt_ok = rt_ok and solved_back
    check("50x scramble + inverse == solved", rt_ok)

    # A scramble is genuinely scrambled (not accidentally solved).
    fac, _ = scramble(n=25, seed=1)
    check("scramble is not solved", fac != SOLVED)

    print(f"\n{'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
