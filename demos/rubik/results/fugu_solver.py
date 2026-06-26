from array import array

SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

_FACE_NAMES = ("U", "R", "F", "D", "L", "B")
_AXIS = (0, 1, 2, 0, 1, 2)

_CORNER_FACELET = (
    (8, 9, 20),    # URF
    (6, 18, 38),   # UFL
    (0, 36, 47),   # ULB
    (2, 45, 11),   # UBR
    (29, 26, 15),  # DFR
    (27, 44, 24),  # DLF
    (33, 53, 42),  # DBL
    (35, 17, 51),  # DRB
)

_EDGE_FACELET = (
    (5, 10),   # UR
    (7, 19),   # UF
    (3, 37),   # UL
    (1, 46),   # UB
    (32, 16),  # DR
    (28, 25),  # DF
    (30, 43),  # DL
    (34, 52),  # DB
    (23, 12),  # FR
    (21, 41),  # FL
    (50, 39),  # BL
    (48, 14),  # BR
)

_CORNER_COLOR = (
    ("U", "R", "F"),
    ("U", "F", "L"),
    ("U", "L", "B"),
    ("U", "B", "R"),
    ("D", "F", "R"),
    ("D", "L", "F"),
    ("D", "B", "L"),
    ("D", "R", "B"),
)

_EDGE_COLOR = (
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

_FACT = [1] * 13
for _i in range(1, 13):
    _FACT[_i] = _FACT[_i - 1] * _i


class _Cubie:
    __slots__ = ("cp", "co", "ep", "eo")

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.cp = list(range(8)) if cp is None else list(cp)
        self.co = [0] * 8 if co is None else list(co)
        self.ep = list(range(12)) if ep is None else list(ep)
        self.eo = [0] * 12 if eo is None else list(eo)


_MOVES = None
_TWIST_MOVE = None
_FLIP_MOVE = None
_SLICE_MOVE = None
_CP_MOVE = None
_EP8_MOVE = None
_SP_MOVE = None

_P1_TS = None
_P1_FS = None
_P2_CS = None
_P2_ES = None

_COMB_IDX = None
_MASKS = None
_SOLVED_SLICE = None
_READY = False

_PHASE2_MOVES = (0, 1, 2, 9, 10, 11, 4, 7, 13, 16)


def _identity():
    return _Cubie()


def _perm_parity(p):
    s = 0
    n = len(p)
    for i in range(n - 1):
        pi = p[i]
        for j in range(i + 1, n):
            if pi > p[j]:
                s ^= 1
    return s


def _facelet_to_cubie(facelet, validate=True):
    if not isinstance(facelet, str) or len(facelet) != 54:
        raise ValueError("facelet string must contain exactly 54 characters")

    if validate:
        allowed = set("URFDLB")
        if any(ch not in allowed for ch in facelet):
            raise ValueError("invalid facelet character")
        if (
            facelet[4] != "U"
            or facelet[13] != "R"
            or facelet[22] != "F"
            or facelet[31] != "D"
            or facelet[40] != "L"
            or facelet[49] != "B"
        ):
            raise ValueError("invalid centres")
        for ch in "URFDLB":
            if facelet.count(ch) != 9:
                raise ValueError("invalid colour count")

    cp = [-1] * 8
    co = [0] * 8
    ep = [-1] * 12
    eo = [0] * 12

    for i in range(8):
        ori = 0
        while ori < 3 and facelet[_CORNER_FACELET[i][ori]] not in ("U", "D"):
            ori += 1
        if ori == 3:
            raise ValueError("invalid corner")

        col1 = facelet[_CORNER_FACELET[i][(ori + 1) % 3]]
        col2 = facelet[_CORNER_FACELET[i][(ori + 2) % 3]]

        found = -1
        for j in range(8):
            if _CORNER_COLOR[j][1] == col1 and _CORNER_COLOR[j][2] == col2:
                found = j
                break
        if found < 0:
            raise ValueError("invalid corner")
        cp[i] = found
        co[i] = ori % 3

    for i in range(12):
        col0 = facelet[_EDGE_FACELET[i][0]]
        col1 = facelet[_EDGE_FACELET[i][1]]
        found = -1
        flip = 0
        for j in range(12):
            if _EDGE_COLOR[j][0] == col0 and _EDGE_COLOR[j][1] == col1:
                found = j
                flip = 0
                break
            if _EDGE_COLOR[j][0] == col1 and _EDGE_COLOR[j][1] == col0:
                found = j
                flip = 1
                break
        if found < 0:
            raise ValueError("invalid edge")
        ep[i] = found
        eo[i] = flip

    if validate:
        if sorted(cp) != list(range(8)) or sorted(ep) != list(range(12)):
            raise ValueError("duplicate cubie")
        if sum(co) % 3 != 0:
            raise ValueError("corner twist error")
        if sum(eo) % 2 != 0:
            raise ValueError("edge flip error")
        if _perm_parity(cp) != _perm_parity(ep):
            raise ValueError("permutation parity error")

    return _Cubie(cp, co, ep, eo)


def _apply_move(c, m):
    cp = c.cp
    co = c.co
    ep = c.ep
    eo = c.eo
    mcp = m.cp
    mco = m.co
    mep = m.ep
    meo = m.eo

    ncp = [0] * 8
    nco = [0] * 8
    for i in range(8):
        j = mcp[i]
        ncp[i] = cp[j]
        nco[i] = (co[j] + mco[i]) % 3

    nep = [0] * 12
    neo = [0] * 12
    for i in range(12):
        j = mep[i]
        nep[i] = ep[j]
        neo[i] = eo[j] ^ meo[i]

    return _Cubie(ncp, nco, nep, neo)


def _apply_sequence(c, seq):
    r = c
    moves = _MOVES
    for m in seq:
        r = _apply_move(r, moves[m])
    return r


def _is_solved(c):
    return (
        c.cp == list(range(8))
        and c.co == [0] * 8
        and c.ep == list(range(12))
        and c.eo == [0] * 12
    )


def _perm_to_idx(p):
    n = len(p)
    idx = 0
    for i in range(n):
        x = p[i]
        less = 0
        for j in range(i + 1, n):
            if p[j] < x:
                less += 1
        idx = idx * (n - i) + less
    return idx


def _idx_to_perm(idx, n):
    elems = list(range(n))
    p = [0] * n
    for i in range(n):
        f = _FACT[n - 1 - i]
        q = idx // f
        idx %= f
        p[i] = elems.pop(q)
    return p


def _get_twist(c):
    r = 0
    co = c.co
    for i in range(7):
        r = 3 * r + co[i]
    return r


def _set_twist(idx):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = idx % 3
        s += co[i]
        idx //= 3
    co[7] = (-s) % 3
    return _Cubie(co=co)


def _get_flip(c):
    r = 0
    eo = c.eo
    for i in range(11):
        r = 2 * r + eo[i]
    return r


def _set_flip(idx):
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = idx & 1
        s += eo[i]
        idx >>= 1
    eo[11] = s & 1
    return _Cubie(eo=eo)


def _get_slice(c):
    mask = 0
    ep = c.ep
    for i in range(12):
        if ep[i] >= 8:
            mask |= 1 << i
    return _COMB_IDX[mask]


def _set_slice(idx):
    mask = _MASKS[idx]
    ep = [0] * 12
    a = 0
    b = 8
    for i in range(12):
        if mask & (1 << i):
            ep[i] = b
            b += 1
        else:
            ep[i] = a
            a += 1
    return _Cubie(ep=ep)


def _get_cp(c):
    return _perm_to_idx(c.cp)


def _set_cp(idx):
    return _Cubie(cp=_idx_to_perm(idx, 8))


def _get_ep8(c):
    return _perm_to_idx(c.ep[:8])


def _set_ep8(idx):
    ep = _idx_to_perm(idx, 8) + [8, 9, 10, 11]
    return _Cubie(ep=ep)


def _get_sp(c):
    return _perm_to_idx([x - 8 for x in c.ep[8:12]])


def _set_sp(idx):
    p = _idx_to_perm(idx, 4)
    ep = list(range(8)) + [x + 8 for x in p]
    return _Cubie(ep=ep)


def _build_combinations():
    global _COMB_IDX, _MASKS, _SOLVED_SLICE

    _COMB_IDX = [-1] * 4096
    _MASKS = []
    for mask in range(4096):
        if mask.bit_count() == 4:
            _COMB_IDX[mask] = len(_MASKS)
            _MASKS.append(mask)
    _SOLVED_SLICE = _get_slice(_identity())


def _rotate_vec(v, axis, sign):
    x, y, z = v
    if axis == 0:
        if sign > 0:
            return (x, -z, y)
        return (x, z, -y)
    if axis == 1:
        if sign > 0:
            return (z, y, -x)
        return (-z, y, x)
    if sign > 0:
        return (-y, x, z)
    return (y, -x, z)


def _facelet_geometry():
    idx_to_key = [None] * 54

    def put(idx, pos, normal):
        idx_to_key[idx] = (pos, normal)

    for r in range(3):
        for c in range(3):
            put(0 + 3 * r + c, (c - 1, 1, r - 1), (0, 1, 0))
            put(9 + 3 * r + c, (1, 1 - r, 1 - c), (1, 0, 0))
            put(18 + 3 * r + c, (c - 1, 1 - r, 1), (0, 0, 1))
            put(27 + 3 * r + c, (c - 1, -1, 1 - r), (0, -1, 0))
            put(36 + 3 * r + c, (-1, 1 - r, c - 1), (-1, 0, 0))
            put(45 + 3 * r + c, (1 - c, 1 - r, -1), (0, 0, -1))

    key_to_idx = {k: i for i, k in enumerate(idx_to_key)}
    return idx_to_key, key_to_idx


def _facelet_perm(face):
    idx_to_key, key_to_idx = _facelet_geometry()

    if face == 0:      # U
        axis, layer, sign = 1, 1, -1
    elif face == 1:    # R
        axis, layer, sign = 0, 1, -1
    elif face == 2:    # F
        axis, layer, sign = 2, 1, -1
    elif face == 3:    # D
        axis, layer, sign = 1, -1, 1
    elif face == 4:    # L
        axis, layer, sign = 0, -1, 1
    else:              # B
        axis, layer, sign = 2, -1, 1

    perm = [0] * 54
    for i, (pos, normal) in enumerate(idx_to_key):
        if pos[axis] == layer:
            npos = _rotate_vec(pos, axis, sign)
            nnormal = _rotate_vec(normal, axis, sign)
        else:
            npos = pos
            nnormal = normal
        perm[i] = key_to_idx[(npos, nnormal)]
    return perm


def _apply_facelet_perm(s, perm):
    a = [""] * 54
    for i, ch in enumerate(s):
        a[perm[i]] = ch
    return "".join(a)


def _build_moves():
    global _MOVES

    quarters = []
    for f in range(6):
        p = _facelet_perm(f)
        moved = _apply_facelet_perm(SOLVED, p)
        quarters.append(_facelet_to_cubie(moved, validate=False))

    moves = []
    for f in range(6):
        q = quarters[f]
        c = _identity()
        for _ in range(3):
            c = _apply_move(c, q)
            moves.append(c)

        check = _apply_move(c, q)
        if not _is_solved(check):
            raise RuntimeError("internal move construction failed")

    _MOVES = moves


def _coord_move_table(size, width, setter, getter, move_indices):
    arr = array("H", [0]) * (size * width)
    moves = _MOVES
    for i in range(size):
        c = setter(i)
        base = i * width
        for j, m in enumerate(move_indices):
            arr[base + j] = getter(_apply_move(c, moves[m]))
    return arr


def _build_pruning(size_a, size_b, table_a, table_b, width, start_a, start_b):
    total = size_a * size_b
    pr = bytearray([255]) * total
    start = start_a * size_b + start_b
    pr[start] = 0

    q = array("I", [start])
    head = 0
    append = q.append
    unknown = 255

    while head < len(q):
        idx = q[head]
        head += 1
        nd = pr[idx] + 1

        a = idx // size_b
        b = idx - a * size_b
        ba = a * width
        bb = b * width

        for m in range(width):
            ni = table_a[ba + m] * size_b + table_b[bb + m]
            if pr[ni] == unknown:
                pr[ni] = nd
                append(ni)

    return pr


def _init():
    global _READY
    global _TWIST_MOVE, _FLIP_MOVE, _SLICE_MOVE, _CP_MOVE, _EP8_MOVE, _SP_MOVE
    global _P1_TS, _P1_FS, _P2_CS, _P2_ES

    if _READY:
        return

    _build_combinations()
    _build_moves()

    all_moves = tuple(range(18))

    _TWIST_MOVE = _coord_move_table(2187, 18, _set_twist, _get_twist, all_moves)
    _FLIP_MOVE = _coord_move_table(2048, 18, _set_flip, _get_flip, all_moves)
    _SLICE_MOVE = _coord_move_table(495, 18, _set_slice, _get_slice, all_moves)

    _CP_MOVE = _coord_move_table(40320, 10, _set_cp, _get_cp, _PHASE2_MOVES)
    _EP8_MOVE = _coord_move_table(40320, 10, _set_ep8, _get_ep8, _PHASE2_MOVES)
    _SP_MOVE = _coord_move_table(24, 10, _set_sp, _get_sp, _PHASE2_MOVES)

    _P1_TS = _build_pruning(2187, 495, _TWIST_MOVE, _SLICE_MOVE, 18, 0, _SOLVED_SLICE)
    _P1_FS = _build_pruning(2048, 495, _FLIP_MOVE, _SLICE_MOVE, 18, 0, _SOLVED_SLICE)

    _P2_CS = _build_pruning(40320, 24, _CP_MOVE, _SP_MOVE, 10, 0, 0)
    _P2_ES = _build_pruning(40320, 24, _EP8_MOVE, _SP_MOVE, 10, 0, 0)

    _READY = True


def _skip_face(face, last_face):
    if last_face < 0:
        return False
    if face == last_face:
        return True
    return _AXIS[face] == _AXIS[last_face] and face < last_face


def _phase2_search(cp, ep, sp, max_depth):
    if max_depth < 0:
        return None

    cp_move = _CP_MOVE
    ep_move = _EP8_MOVE
    sp_move = _SP_MOVE
    pr_cs = _P2_CS
    pr_es = _P2_ES
    moves = _PHASE2_MOVES
    path = []

    h0 = pr_cs[cp * 24 + sp]
    h1 = pr_es[ep * 24 + sp]
    h = h0 if h0 > h1 else h1
    if h == 255 or h > max_depth:
        return None

    def dfs(cpc, epc, spc, depth, last_face):
        a = pr_cs[cpc * 24 + spc]
        b = pr_es[epc * 24 + spc]
        if a > depth or b > depth:
            return False
        if depth == 0:
            return cpc == 0 and epc == 0 and spc == 0

        bcp = cpc * 10
        bep = epc * 10
        bsp = spc * 10

        for mi, gm in enumerate(moves):
            face = gm // 3
            if _skip_face(face, last_face):
                continue

            path.append(gm)
            if dfs(
                cp_move[bcp + mi],
                ep_move[bep + mi],
                sp_move[bsp + mi],
                depth - 1,
                face,
            ):
                return True
            path.pop()

        return False

    for depth in range(h, max_depth + 1):
        path.clear()
        if dfs(cp, ep, sp, depth, -1):
            return list(path)
    return None


def _normalize(seq):
    out = []
    for m in seq:
        f = m // 3
        p = (m % 3) + 1
        if out and out[-1] // 3 == f:
            old = (out[-1] % 3) + 1
            new = (old + p) % 4
            if new == 0:
                out.pop()
            else:
                out[-1] = f * 3 + (new - 1)
        else:
            out.append(m)
    return out


def _seq_to_string(seq):
    parts = []
    for m in seq:
        f = m // 3
        p = m % 3
        if p == 0:
            parts.append(_FACE_NAMES[f])
        elif p == 1:
            parts.append(_FACE_NAMES[f] + "2")
        else:
            parts.append(_FACE_NAMES[f] + "'")
    return " ".join(parts)


def _solve_cubie(cube):
    tw = _get_twist(cube)
    fl = _get_flip(cube)
    sl = _get_slice(cube)

    best = None
    best_len = 999
    tried = 0
    stop = False
    path = []

    tw_move = _TWIST_MOVE
    fl_move = _FLIP_MOVE
    sl_move = _SLICE_MOVE
    pr_ts = _P1_TS
    pr_fs = _P1_FS
    solved_slice = _SOLVED_SLICE

    h0 = pr_ts[tw * 495 + sl]
    h1 = pr_fs[fl * 495 + sl]
    start_depth = h0 if h0 > h1 else h1
    if start_depth == 255:
        raise ValueError("unsolvable cube")

    def handle_phase1_solution():
        nonlocal best, best_len, tried, stop

        tried += 1
        mid = _apply_sequence(cube, path)

        cp = _get_cp(mid)
        ep = _get_ep8(mid)
        sp = _get_sp(mid)

        p2_max = min(18, best_len - len(path) + 1)
        p2 = _phase2_search(cp, ep, sp, p2_max)
        if p2 is not None:
            seq = _normalize(path + p2)
            if len(seq) < best_len:
                solved_check = _apply_sequence(cube, seq)
                if _is_solved(solved_check):
                    best = seq
                    best_len = len(seq)

        if best is not None:
            if best_len <= 20:
                stop = True
            elif tried >= 80 and best_len <= 23:
                stop = True
            elif tried >= 300:
                stop = True

        return stop

    def dfs1(twc, flc, slc, depth, last_face):
        a = pr_ts[twc * 495 + slc]
        b = pr_fs[flc * 495 + slc]
        if a > depth or b > depth:
            return False

        if depth == 0:
            if twc == 0 and flc == 0 and slc == solved_slice:
                return handle_phase1_solution()
            return False

        btw = twc * 18
        bfl = flc * 18
        bsl = slc * 18

        for m in range(18):
            face = m // 3
            if _skip_face(face, last_face):
                continue

            path.append(m)
            if dfs1(
                tw_move[btw + m],
                fl_move[bfl + m],
                sl_move[bsl + m],
                depth - 1,
                face,
            ):
                return True
            path.pop()

        return False

    max_phase1 = 12
    for depth in range(start_depth, max_phase1 + 1):
        path.clear()
        if dfs1(tw, fl, sl, depth, -1):
            break
        if stop:
            break

    if best is None:
        for depth in range(max_phase1 + 1, 14):
            path.clear()
            if dfs1(tw, fl, sl, depth, -1):
                break
            if best is not None:
                break

    if best is None:
        raise RuntimeError("no solution found")

    return _seq_to_string(best)


def solve(facelet: str) -> str:
    if facelet == SOLVED:
        return ""

    cube = _facelet_to_cubie(facelet, validate=True)
    if _is_solved(cube):
        return ""

    _init()
    return _solve_cubie(cube)