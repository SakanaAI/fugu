from array import array
from itertools import combinations

_FACE_NAMES = ("U", "R", "F", "D", "L", "B")
_SUFFIXES = ("", "2", "'")
_TOKENS = tuple(f + s for f in _FACE_NAMES for s in _SUFFIXES)
_FACE_OF_MOVE = tuple(i // 3 for i in range(18))
_POWER_OF_MOVE = tuple((1, 2, 3)[i % 3] for i in range(18))
_OPP = (3, 4, 5, 0, 1, 2)
_P2_MOVES = (0, 1, 2, 9, 10, 11, 4, 7, 13, 16)
_P2_FACES = tuple(_FACE_OF_MOVE[m] for m in _P2_MOVES)

_CORNER_FACELETS = (
    (8, 9, 20),
    (6, 18, 38),
    (0, 36, 47),
    (2, 45, 11),
    (29, 26, 15),
    (27, 44, 24),
    (33, 53, 42),
    (35, 17, 51),
)
_CORNER_COLORS = (
    ("U", "R", "F"),
    ("U", "F", "L"),
    ("U", "L", "B"),
    ("U", "B", "R"),
    ("D", "F", "R"),
    ("D", "L", "F"),
    ("D", "B", "L"),
    ("D", "R", "B"),
)
_EDGE_FACELETS = (
    (5, 10),
    (7, 19),
    (3, 37),
    (1, 46),
    (32, 16),
    (28, 25),
    (30, 43),
    (34, 52),
    (23, 12),
    (21, 41),
    (50, 39),
    (48, 14),
)
_EDGE_COLORS = (
    ("U", "R"),
    ("U", "F"),
    ("U", "L"),
    ("U", "B"),
    ("D", "R"),
    ("D", "F"),
    ("D", "L"),
    ("D", "B"),
    ("F", "R"),
    ("F", "L"),
    ("B", "L"),
    ("B", "R"),
)

_BASE_CP = (
    (3, 0, 1, 2, 4, 5, 6, 7),
    (4, 1, 2, 0, 7, 5, 6, 3),
    (1, 5, 2, 3, 0, 4, 6, 7),
    (0, 1, 2, 3, 5, 6, 7, 4),
    (0, 2, 6, 3, 4, 1, 5, 7),
    (0, 1, 3, 7, 4, 5, 2, 6),
)
_BASE_CO = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 1, 1, 0, 0, 2),
    (1, 2, 0, 0, 2, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 2, 0, 0, 2, 1, 0),
    (0, 0, 1, 2, 0, 0, 2, 1),
)
_BASE_EP = (
    (3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
    (8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0),
    (0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11),
    (0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11),
    (0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11),
    (0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7),
)
_BASE_EO = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1),
)

_FACT = [1]
for _i in range(1, 13):
    _FACT.append(_FACT[-1] * _i)

_COMBOS = []
_COMB_INDEX = {}
for _c in combinations(range(12), 4):
    _m = 0
    for _x in _c:
        _m |= 1 << _x
    _COMB_INDEX[_m] = len(_COMBOS)
    _COMBOS.append(_m)
_SOLVED_SLICE = _COMB_INDEX[(1 << 8) | (1 << 9) | (1 << 10) | (1 << 11)]

_MOVE_CP = []
_MOVE_CO = []
_MOVE_EP = []
_MOVE_EO = []


def _compose(cp, co, ep, eo, bcp, bco, bep, beo):
    ncp = [0] * 8
    nco = [0] * 8
    nep = [0] * 12
    neo = [0] * 12
    for i in range(8):
        j = bcp[i]
        ncp[i] = cp[j]
        nco[i] = (co[j] + bco[i]) % 3
    for i in range(12):
        j = bep[i]
        nep[i] = ep[j]
        neo[i] = eo[j] ^ beo[i]
    return ncp, nco, nep, neo


for _f in range(6):
    _cp = list(range(8))
    _co = [0] * 8
    _ep = list(range(12))
    _eo = [0] * 12
    for _p in range(3):
        _cp, _co, _ep, _eo = _compose(
            _cp, _co, _ep, _eo,
            _BASE_CP[_f], _BASE_CO[_f], _BASE_EP[_f], _BASE_EO[_f]
        )
        _MOVE_CP.append(tuple(_cp))
        _MOVE_CO.append(tuple(_co))
        _MOVE_EP.append(tuple(_ep))
        _MOVE_EO.append(tuple(_eo))

_TABLES_READY = False
_TWIST_MOVE = None
_FLIP_MOVE = None
_SLICE_MOVE = None
_CP_MOVE = None
_UD_MOVE = None
_SP_MOVE = None
_PRUNE_TWIST_SLICE = None
_PRUNE_FLIP_SLICE = None
_PRUNE_CP_SP = None
_PRUNE_UD_SP = None


def solve(facelet: str) -> str:
    cp0, co0, ep0, eo0 = _facelet_to_cubie(facelet)
    _verify_cubie(cp0, co0, ep0, eo0)

    if (
        cp0 == list(range(8))
        and ep0 == list(range(12))
        and not any(co0)
        and not any(eo0)
    ):
        return ""

    _init_tables()

    twist0 = _twist_to_index(co0)
    flip0 = _flip_to_index(eo0)
    slice0 = _get_slice(ep0)

    twist_move = _TWIST_MOVE
    flip_move = _FLIP_MOVE
    slice_move = _SLICE_MOVE
    cp_move = _CP_MOVE
    ud_move = _UD_MOVE
    sp_move = _SP_MOVE
    prune_ts = _PRUNE_TWIST_SLICE
    prune_fs = _PRUNE_FLIP_SLICE
    prune_cp = _PRUNE_CP_SP
    prune_ud = _PRUNE_UD_SP

    face_of = _FACE_OF_MOVE
    opp = _OPP
    p2_moves = _P2_MOVES
    p2_faces = _P2_FACES

    h10 = max(
        prune_ts[twist0 * 495 + slice0],
        prune_fs[flip0 * 495 + slice0],
    )

    def phase2(cp_idx, ud_idx, sp_idx, max_depth, last_face):
        if max_depth < 0:
            return None
        if max_depth > 18:
            max_depth = 18

        h0 = max(prune_cp[cp_idx * 24 + sp_idx], prune_ud[ud_idx * 24 + sp_idx])
        if h0 > max_depth:
            return None

        path2 = []

        def dfs2(c, u, s, depth, last):
            if c == 0 and u == 0 and s == 0:
                return True
            if depth == 0:
                return False

            rem = depth - 1
            for k, face in enumerate(p2_faces):
                if last >= 0 and (face == last or (opp[face] == last and face < last)):
                    continue

                nc = cp_move[k][c]
                nu = ud_move[k][u]
                ns = sp_move[k][s]
                if max(prune_cp[nc * 24 + ns], prune_ud[nu * 24 + ns]) > rem:
                    continue

                path2.append(p2_moves[k])
                if dfs2(nc, nu, ns, rem, face):
                    return True
                path2.pop()

            return False

        for depth in range(h0, max_depth + 1):
            if dfs2(cp_idx, ud_idx, sp_idx, depth, last_face):
                return path2[:]
        return None

    def phase2_indices_after_path(path):
        cp = cp0[:]
        ep = ep0[:]
        for m in path:
            mcp = _MOVE_CP[m]
            mep = _MOVE_EP[m]
            cp = [cp[mcp[i]] for i in range(8)]
            ep = [ep[mep[i]] for i in range(12)]
        return (
            _perm_to_index(cp),
            _perm_to_index(ep[:8]),
            _perm_to_index([ep[8] - 8, ep[9] - 8, ep[10] - 8, ep[11] - 8]),
        )

    def dfs1(tw, fl, sl, depth, last_face, path, total_limit, fixed_p2_limit):
        if depth == 0:
            if tw == 0 and fl == 0 and sl == _SOLVED_SLICE:
                c2, u2, s2 = phase2_indices_after_path(path)
                if total_limit is None:
                    p2_limit = fixed_p2_limit
                else:
                    p2_limit = total_limit - len(path)
                p2 = phase2(c2, u2, s2, p2_limit, last_face)
                if p2 is not None:
                    return path[:] + p2
            return None

        rem = depth - 1
        for m in range(18):
            face = face_of[m]
            if last_face >= 0 and (face == last_face or (opp[face] == last_face and face < last_face)):
                continue

            nt = twist_move[m][tw]
            nf = flip_move[m][fl]
            ns = slice_move[m][sl]

            if max(prune_ts[nt * 495 + ns], prune_fs[nf * 495 + ns]) > rem:
                continue

            path.append(m)
            r = dfs1(nt, nf, ns, rem, face, path, total_limit, fixed_p2_limit)
            if r is not None:
                return r
            path.pop()

        return None

    def run_phase1(depth1, total_limit, fixed_p2_limit):
        if h10 > depth1:
            return None
        path = []
        return dfs1(twist0, flip0, slice0, depth1, -1, path, total_limit, fixed_p2_limit)

    max_total = 24
    max_phase1 = 12

    for total in range(int(h10), max_total + 1):
        dmin = max(int(h10), total - 18)
        dmax = min(max_phase1, total)
        for d1 in range(dmin, dmax + 1):
            r = run_phase1(d1, total, None)
            if r is not None:
                return _format_solution(_simplify(r))

    for d1 in range(int(h10), max_phase1 + 1):
        r = run_phase1(d1, None, 18)
        if r is not None:
            return _format_solution(_simplify(r))

    raise RuntimeError("no solution found")


def _init_tables():
    global _TABLES_READY
    global _TWIST_MOVE, _FLIP_MOVE, _SLICE_MOVE
    global _CP_MOVE, _UD_MOVE, _SP_MOVE
    global _PRUNE_TWIST_SLICE, _PRUNE_FLIP_SLICE, _PRUNE_CP_SP, _PRUNE_UD_SP

    if _TABLES_READY:
        return

    _TWIST_MOVE = _build_twist_move()
    _FLIP_MOVE = _build_flip_move()
    _SLICE_MOVE = _build_slice_move()

    _CP_MOVE = _build_cp_move()
    _UD_MOVE = _build_ud_move()
    _SP_MOVE = _build_sp_move()

    _PRUNE_TWIST_SLICE = _build_prune(2187, 495, _TWIST_MOVE, _SLICE_MOVE, 0, _SOLVED_SLICE)
    _PRUNE_FLIP_SLICE = _build_prune(2048, 495, _FLIP_MOVE, _SLICE_MOVE, 0, _SOLVED_SLICE)
    _PRUNE_CP_SP = _build_prune(40320, 24, _CP_MOVE, _SP_MOVE, 0, 0)
    _PRUNE_UD_SP = _build_prune(40320, 24, _UD_MOVE, _SP_MOVE, 0, 0)

    _TABLES_READY = True


def _build_prune(n1, n2, move1, move2, start1, start2):
    size = n1 * n2
    dist = bytearray([255]) * size
    start = start1 * n2 + start2
    dist[start] = 0
    frontier = [start]
    pairs = tuple(zip(move1, move2))
    depth = 0

    while frontier:
        nd = depth + 1
        nxt = []
        for idx in frontier:
            a = idx // n2
            b = idx - a * n2
            for ma, mb in pairs:
                ni = ma[a] * n2 + mb[b]
                if dist[ni] == 255:
                    dist[ni] = nd
                    nxt.append(ni)
        frontier = nxt
        depth = nd

    return dist


def _build_twist_move():
    tabs = [array("H", [0]) * 2187 for _ in range(18)]
    for idx in range(2187):
        co = _index_to_twist(idx)
        for m in range(18):
            mcp = _MOVE_CP[m]
            mco = _MOVE_CO[m]
            nco = [0] * 8
            for i in range(8):
                nco[i] = (co[mcp[i]] + mco[i]) % 3
            tabs[m][idx] = _twist_to_index(nco)
    return tuple(tabs)


def _build_flip_move():
    tabs = [array("H", [0]) * 2048 for _ in range(18)]
    for idx in range(2048):
        eo = _index_to_flip(idx)
        for m in range(18):
            mep = _MOVE_EP[m]
            meo = _MOVE_EO[m]
            neo = [0] * 12
            for i in range(12):
                neo[i] = eo[mep[i]] ^ meo[i]
            tabs[m][idx] = _flip_to_index(neo)
    return tuple(tabs)


def _build_slice_move():
    tabs = [array("H", [0]) * 495 for _ in range(18)]
    for idx in range(495):
        ep = _index_to_slice_edges(idx)
        for m in range(18):
            mep = _MOVE_EP[m]
            nep = [ep[mep[i]] for i in range(12)]
            tabs[m][idx] = _get_slice(nep)
    return tuple(tabs)


def _build_cp_move():
    tabs = [array("H", [0]) * 40320 for _ in _P2_MOVES]
    for idx in range(40320):
        p = _index_to_perm(8, idx)
        for k, m in enumerate(_P2_MOVES):
            mcp = _MOVE_CP[m]
            np = [p[mcp[i]] for i in range(8)]
            tabs[k][idx] = _perm_to_index(np)
    return tuple(tabs)


def _build_ud_move():
    tabs = [array("H", [0]) * 40320 for _ in _P2_MOVES]
    for idx in range(40320):
        p = _index_to_perm(8, idx)
        for k, m in enumerate(_P2_MOVES):
            mep = _MOVE_EP[m]
            np = [p[mep[i]] for i in range(8)]
            tabs[k][idx] = _perm_to_index(np)
    return tuple(tabs)


def _build_sp_move():
    tabs = [array("H", [0]) * 24 for _ in _P2_MOVES]
    for idx in range(24):
        p = _index_to_perm(4, idx)
        for k, m in enumerate(_P2_MOVES):
            mep = _MOVE_EP[m]
            np = [p[mep[8 + i] - 8] for i in range(4)]
            tabs[k][idx] = _perm_to_index(np)
    return tuple(tabs)


def _twist_to_index(co):
    idx = 0
    for i in range(7):
        idx = idx * 3 + co[i]
    return idx


def _index_to_twist(idx):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = idx % 3
        s += co[i]
        idx //= 3
    co[7] = (-s) % 3
    return co


def _flip_to_index(eo):
    idx = 0
    for i in range(11):
        idx = idx * 2 + eo[i]
    return idx


def _index_to_flip(idx):
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = idx & 1
        s ^= eo[i]
        idx >>= 1
    eo[11] = s
    return eo


def _get_slice(ep):
    mask = 0
    for i, e in enumerate(ep):
        if e >= 8:
            mask |= 1 << i
    return _COMB_INDEX[mask]


def _index_to_slice_edges(idx):
    mask = _COMBOS[idx]
    ep = [0] * 12
    a = 0
    b = 8
    for i in range(12):
        if (mask >> i) & 1:
            ep[i] = b
            b += 1
        else:
            ep[i] = a
            a += 1
    return ep


def _perm_to_index(p):
    n = len(p)
    idx = 0
    for i in range(n - 1):
        c = 0
        pi = p[i]
        for j in range(i + 1, n):
            if p[j] < pi:
                c += 1
        idx += c * _FACT[n - 1 - i]
    return idx


def _index_to_perm(n, idx):
    elems = list(range(n))
    p = [0] * n
    for i in range(n):
        f = _FACT[n - 1 - i]
        q = idx // f
        idx %= f
        p[i] = elems.pop(q)
    return p


def _facelet_to_cubie(s):
    if not isinstance(s, str) or len(s) != 54:
        raise ValueError("facelet must be a 54-character string")

    allowed = set("URFDLB")
    if any(ch not in allowed for ch in s):
        raise ValueError("invalid facelet character")

    for ch in "URFDLB":
        if s.count(ch) != 9:
            raise ValueError("invalid colour count")

    for idx, ch in ((4, "U"), (13, "R"), (22, "F"), (31, "D"), (40, "L"), (49, "B")):
        if s[idx] != ch:
            raise ValueError("invalid centre facelet")

    cp = [-1] * 8
    co = [0] * 8
    for i in range(8):
        cols = [s[x] for x in _CORNER_FACELETS[i]]
        ori = -1
        for k in range(3):
            if cols[k] == "U" or cols[k] == "D":
                ori = k
                break
        if ori < 0:
            raise ValueError("invalid corner")

        c1 = cols[(ori + 1) % 3]
        c2 = cols[(ori + 2) % 3]
        found = -1
        for j in range(8):
            if _CORNER_COLORS[j][1] == c1 and _CORNER_COLORS[j][2] == c2:
                found = j
                break
        if found < 0:
            raise ValueError("invalid corner")
        cp[i] = found
        co[i] = ori % 3

    ep = [-1] * 12
    eo = [0] * 12
    for i in range(12):
        c0 = s[_EDGE_FACELETS[i][0]]
        c1 = s[_EDGE_FACELETS[i][1]]
        found = -1
        flip = 0
        for j in range(12):
            if _EDGE_COLORS[j][0] == c0 and _EDGE_COLORS[j][1] == c1:
                found = j
                flip = 0
                break
            if _EDGE_COLORS[j][0] == c1 and _EDGE_COLORS[j][1] == c0:
                found = j
                flip = 1
                break
        if found < 0:
            raise ValueError("invalid edge")
        ep[i] = found
        eo[i] = flip

    return cp, co, ep, eo


def _verify_cubie(cp, co, ep, eo):
    if sorted(cp) != list(range(8)):
        raise ValueError("invalid corner permutation")
    if sorted(ep) != list(range(12)):
        raise ValueError("invalid edge permutation")
    if sum(co) % 3 != 0:
        raise ValueError("invalid corner orientation")
    if sum(eo) % 2 != 0:
        raise ValueError("invalid edge orientation")
    if _parity(cp) != _parity(ep):
        raise ValueError("invalid permutation parity")


def _parity(p):
    r = 0
    n = len(p)
    for i in range(n - 1):
        pi = p[i]
        for j in range(i + 1, n):
            if p[j] < pi:
                r ^= 1
    return r


def _simplify(moves):
    out = []
    pow_to_offset = {1: 0, 2: 1, 3: 2}
    for m in moves:
        f = _FACE_OF_MOVE[m]
        p = _POWER_OF_MOVE[m]
        if out and _FACE_OF_MOVE[out[-1]] == f:
            q = (_POWER_OF_MOVE[out[-1]] + p) % 4
            out.pop()
            if q:
                out.append(f * 3 + pow_to_offset[q])
        else:
            out.append(m)
    return out


def _format_solution(moves):
    return " ".join(_TOKENS[m] for m in moves)