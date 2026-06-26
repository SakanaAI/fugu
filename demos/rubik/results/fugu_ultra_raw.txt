from itertools import combinations, permutations
from array import array

_FACE_NAMES = "URFDLB"
_FACE_SET = set(_FACE_NAMES)
_SOLVED = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

_SOLVED_CP = list(range(8))
_SOLVED_CO = [0] * 8
_SOLVED_EP = list(range(12))
_SOLVED_EO = [0] * 12

_CORNER_FACELET = (
    (8, 9, 20), (6, 18, 38), (0, 36, 47), (2, 45, 11),
    (29, 26, 15), (27, 44, 24), (33, 53, 42), (35, 17, 51),
)
_CORNER_COLOR = (
    ("U", "R", "F"), ("U", "F", "L"), ("U", "L", "B"), ("U", "B", "R"),
    ("D", "F", "R"), ("D", "L", "F"), ("D", "B", "L"), ("D", "R", "B"),
)

_EDGE_FACELET = (
    (5, 10), (7, 19), (3, 37), (1, 46),
    (32, 16), (28, 25), (30, 43), (34, 52),
    (23, 12), (21, 41), (50, 39), (48, 14),
)
_EDGE_COLOR = (
    ("U", "R"), ("U", "F"), ("U", "L"), ("U", "B"),
    ("D", "R"), ("D", "F"), ("D", "L"), ("D", "B"),
    ("F", "R"), ("F", "L"), ("B", "L"), ("B", "R"),
)

_NTWIST = 2187
_NFLIP = 2048

_COMBOS = tuple(combinations(range(12), 4))
_COMB_RANK = {c: i for i, c in enumerate(_COMBOS)}
_NSLICE = len(_COMBOS)
_SLICE_GOAL = _COMB_RANK[(8, 9, 10, 11)]

_PERM8 = tuple(permutations(range(8)))
_PERM8_RANK = {p: i for i, p in enumerate(_PERM8)}
_NPERM8 = len(_PERM8)

_PERM4 = tuple(permutations(range(4)))
_PERM4_RANK = {p: i for i, p in enumerate(_PERM4)}
_NPERM4 = len(_PERM4)

_MOVE_NAMES = tuple(f + s for f in _FACE_NAMES for s in ("", "2", "'"))
_P1_MOVE_IDS = tuple(range(18))
_P2_MOVE_IDS = (0, 1, 2, 9, 10, 11, 4, 7, 13, 16)
_P2_FACES = tuple(m // 3 for m in _P2_MOVE_IDS)
_NP2 = len(_P2_MOVE_IDS)

_AXIS_OF_FACE = (0, 1, 2, 0, 1, 2)
_AXIS_FACES = ((0, 3), (1, 4), (2, 5))
_POWER_OF = (1, 2, 3)
_ID_OF_POWER = {1: 0, 2: 1, 3: 2}

_PHASE1_MAX = 12
_PHASE2_MAX = 18
_TOTAL_MAX = 30

_MOVE_CUBES = None
_P1_TWIST_MOVE = None
_P1_FLIP_MOVE = None
_P1_SLICE_MOVE = None
_P2_CP_MOVE = None
_P2_UD_MOVE = None
_P2_SP_MOVE = None
_P1_TWIST_SLICE = None
_P1_FLIP_SLICE = None
_P2_CP_SLICE = None
_P2_UD_SLICE = None
_INITIALIZED = False


class _CubieCube:
    __slots__ = ("cp", "co", "ep", "eo")

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.cp = list(_SOLVED_CP) if cp is None else list(cp)
        self.co = list(_SOLVED_CO) if co is None else list(co)
        self.ep = list(_SOLVED_EP) if ep is None else list(ep)
        self.eo = list(_SOLVED_EO) if eo is None else list(eo)


def _multiply(a, b):
    acp, aco, aep, aeo = a.cp, a.co, a.ep, a.eo
    bcp, bco, bep, beo = b.cp, b.co, b.ep, b.eo

    cp = [0] * 8
    co = [0] * 8
    for i in range(8):
        j = bcp[i]
        cp[i] = acp[j]
        co[i] = (aco[j] + bco[i]) % 3

    ep = [0] * 12
    eo = [0] * 12
    for i in range(12):
        j = bep[i]
        ep[i] = aep[j]
        eo[i] = aeo[j] ^ beo[i]

    return _CubieCube(cp, co, ep, eo)


def _rotate_vec(v, axis, turn):
    x, y, z = v
    if axis == 0:
        if turn == 1:
            return (x, -z, y)
        return (x, z, -y)
    if axis == 1:
        if turn == 1:
            return (z, y, -x)
        return (-z, y, x)
    if turn == 1:
        return (-y, x, z)
    return (y, -x, z)


def _facelet_permutations():
    geom = [None] * 54
    index = {}

    def add(i, pos, normal):
        geom[i] = (pos, normal)
        index[(pos, normal)] = i

    for r in range(3):
        for c in range(3):
            add(0 + 3 * r + c, (c - 1, 1, r - 1), (0, 1, 0))
            add(9 + 3 * r + c, (1, 1 - r, 1 - c), (1, 0, 0))
            add(18 + 3 * r + c, (c - 1, 1 - r, 1), (0, 0, 1))
            add(27 + 3 * r + c, (c - 1, -1, 1 - r), (0, -1, 0))
            add(36 + 3 * r + c, (-1, 1 - r, c - 1), (-1, 0, 0))
            add(45 + 3 * r + c, (1 - c, 1 - r, -1), (0, 0, -1))

    face_axis_sign = ((1, 1), (0, 1), (2, 1), (1, -1), (0, -1), (2, -1))
    perms = []
    for axis, sign in face_axis_sign:
        turn = -sign
        perm = list(range(54))
        for src, (pos, normal) in enumerate(geom):
            if pos[axis] == sign:
                npos = _rotate_vec(pos, axis, turn)
                nnorm = _rotate_vec(normal, axis, turn)
                dst = index[(npos, nnorm)]
                perm[dst] = src
        perms.append(tuple(perm))
    return tuple(perms)


def _apply_facelet_perm(s, p):
    return "".join(s[p[i]] for i in range(54))


def _cubie_from_facelets(facelet):
    cp = [-1] * 8
    co = [0] * 8
    for i in range(8):
        ori = -1
        for k in range(3):
            c = facelet[_CORNER_FACELET[i][k]]
            if c == "U" or c == "D":
                ori = k
                break
        if ori < 0:
            raise ValueError("invalid cube")
        c1 = facelet[_CORNER_FACELET[i][(ori + 1) % 3]]
        c2 = facelet[_CORNER_FACELET[i][(ori + 2) % 3]]
        found = -1
        for j in range(8):
            if c1 == _CORNER_COLOR[j][1] and c2 == _CORNER_COLOR[j][2]:
                found = j
                break
        if found < 0:
            raise ValueError("invalid cube")
        cp[i] = found
        co[i] = ori % 3

    ep = [-1] * 12
    eo = [0] * 12
    for i in range(12):
        a = facelet[_EDGE_FACELET[i][0]]
        b = facelet[_EDGE_FACELET[i][1]]
        found = -1
        ori = 0
        for j in range(12):
            if a == _EDGE_COLOR[j][0] and b == _EDGE_COLOR[j][1]:
                found = j
                ori = 0
                break
            if a == _EDGE_COLOR[j][1] and b == _EDGE_COLOR[j][0]:
                found = j
                ori = 1
                break
        if found < 0:
            raise ValueError("invalid cube")
        ep[i] = found
        eo[i] = ori

    return _CubieCube(cp, co, ep, eo)


def _build_move_cubes():
    quarters = []
    for p in _facelet_permutations():
        quarters.append(_cubie_from_facelets(_apply_facelet_perm(_SOLVED, p)))

    moves = [None] * 18
    ident = _CubieCube()
    for face in range(6):
        cur = ident
        q = quarters[face]
        for power in range(3):
            cur = _multiply(cur, q)
            moves[3 * face + power] = cur
    return tuple(moves)


def _get_twist(c):
    idx = 0
    co = c.co
    for i in range(7):
        idx = 3 * idx + co[i]
    return idx


def _cube_from_twist(idx):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = idx % 3
        s += co[i]
        idx //= 3
    co[7] = (-s) % 3
    return _CubieCube(co=co)


def _get_flip(c):
    idx = 0
    eo = c.eo
    for i in range(11):
        idx = 2 * idx + eo[i]
    return idx


def _cube_from_flip(idx):
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = idx & 1
        s += eo[i]
        idx >>= 1
    eo[11] = s & 1
    return _CubieCube(eo=eo)


def _get_slice_combo(c):
    return _COMB_RANK[tuple(i for i, e in enumerate(c.ep) if e >= 8)]


def _cube_from_slice_combo(idx):
    ep = [-1] * 12
    combo = _COMBOS[idx]
    for k, pos in enumerate(combo):
        ep[pos] = 8 + k
    n = 0
    for i in range(12):
        if ep[i] < 0:
            ep[i] = n
            n += 1
    return _CubieCube(ep=ep)


def _get_cp(c):
    return _PERM8_RANK[tuple(c.cp)]


def _cube_from_cp(idx):
    return _CubieCube(cp=_PERM8[idx])


def _get_ud_edges(c):
    return _PERM8_RANK[tuple(c.ep[:8])]


def _cube_from_ud_edges(idx):
    return _CubieCube(ep=list(_PERM8[idx]) + [8, 9, 10, 11])


def _get_slice_perm(c):
    return _PERM4_RANK[tuple(e - 8 for e in c.ep[8:12])]


def _cube_from_slice_perm(idx):
    return _CubieCube(ep=list(range(8)) + [8 + x for x in _PERM4[idx]])


def _build_move_table(ncoord, nmoves, make_cube, get_coord, move_ids):
    tab = array("H", [0]) * (ncoord * nmoves)
    moves = _MOVE_CUBES
    for i in range(ncoord):
        c = make_cube(i)
        base = i * nmoves
        for j, m in enumerate(move_ids):
            tab[base + j] = get_coord(_multiply(c, moves[m]))
    return tab


def _build_prune(size_a, size_b, move_a, move_b, nmoves, goal_a, goal_b):
    total = size_a * size_b
    pr = bytearray([255]) * total
    start = goal_a * size_b + goal_b
    pr[start] = 0
    q = [start]
    head = 0

    while head < len(q):
        idx = q[head]
        head += 1
        nd = pr[idx] + 1
        a = idx // size_b
        b = idx - a * size_b
        ba = a * nmoves
        bb = b * nmoves
        for m in range(nmoves):
            ni = move_a[ba + m] * size_b + move_b[bb + m]
            if pr[ni] == 255:
                pr[ni] = nd
                q.append(ni)
    return pr


def _init():
    global _INITIALIZED, _MOVE_CUBES
    global _P1_TWIST_MOVE, _P1_FLIP_MOVE, _P1_SLICE_MOVE
    global _P2_CP_MOVE, _P2_UD_MOVE, _P2_SP_MOVE
    global _P1_TWIST_SLICE, _P1_FLIP_SLICE, _P2_CP_SLICE, _P2_UD_SLICE

    if _INITIALIZED:
        return

    _MOVE_CUBES = _build_move_cubes()

    _P1_TWIST_MOVE = _build_move_table(_NTWIST, 18, _cube_from_twist, _get_twist, _P1_MOVE_IDS)
    _P1_FLIP_MOVE = _build_move_table(_NFLIP, 18, _cube_from_flip, _get_flip, _P1_MOVE_IDS)
    _P1_SLICE_MOVE = _build_move_table(_NSLICE, 18, _cube_from_slice_combo, _get_slice_combo, _P1_MOVE_IDS)

    _P2_CP_MOVE = _build_move_table(_NPERM8, _NP2, _cube_from_cp, _get_cp, _P2_MOVE_IDS)
    _P2_UD_MOVE = _build_move_table(_NPERM8, _NP2, _cube_from_ud_edges, _get_ud_edges, _P2_MOVE_IDS)
    _P2_SP_MOVE = _build_move_table(_NPERM4, _NP2, _cube_from_slice_perm, _get_slice_perm, _P2_MOVE_IDS)

    _P1_TWIST_SLICE = _build_prune(_NTWIST, _NSLICE, _P1_TWIST_MOVE, _P1_SLICE_MOVE, 18, 0, _SLICE_GOAL)
    _P1_FLIP_SLICE = _build_prune(_NFLIP, _NSLICE, _P1_FLIP_MOVE, _P1_SLICE_MOVE, 18, 0, _SLICE_GOAL)
    _P2_CP_SLICE = _build_prune(_NPERM8, _NPERM4, _P2_CP_MOVE, _P2_SP_MOVE, _NP2, 0, 0)
    _P2_UD_SLICE = _build_prune(_NPERM8, _NPERM4, _P2_UD_MOVE, _P2_SP_MOVE, _NP2, 0, 0)

    _INITIALIZED = True


def _skip_phase1(face, last_face):
    if last_face < 0:
        return False
    if face == last_face:
        return True
    return _AXIS_OF_FACE[face] == _AXIS_OF_FACE[last_face] and face < last_face


def _skip_phase2(face, last_face, first):
    if last_face < 0:
        return False
    if face == last_face:
        return True
    return (not first) and _AXIS_OF_FACE[face] == _AXIS_OF_FACE[last_face] and face < last_face


def _apply_path(c, path):
    r = c
    moves = _MOVE_CUBES
    for m in path:
        r = _multiply(r, moves[m])
    return r


def _is_solved(c):
    return (
        c.cp == _SOLVED_CP and c.co == _SOLVED_CO and
        c.ep == _SOLVED_EP and c.eo == _SOLVED_EO
    )


def _phase2_dfs(cp, ud, sp, depth_left, last_face, first, path):
    h1 = _P2_CP_SLICE[cp * _NPERM4 + sp]
    if h1 > depth_left:
        return None
    h2 = _P2_UD_SLICE[ud * _NPERM4 + sp]
    if h2 > depth_left:
        return None

    if depth_left == 0:
        if cp == 0 and ud == 0 and sp == 0:
            return list(path)
        return None

    bcp = cp * _NP2
    bud = ud * _NP2
    bsp = sp * _NP2

    for i, mid in enumerate(_P2_MOVE_IDS):
        face = _P2_FACES[i]
        if _skip_phase2(face, last_face, first):
            continue

        ncp = _P2_CP_MOVE[bcp + i]
        nud = _P2_UD_MOVE[bud + i]
        nsp = _P2_SP_MOVE[bsp + i]

        path.append(mid)
        r = _phase2_dfs(ncp, nud, nsp, depth_left - 1, face, False, path)
        if r is not None:
            return r
        path.pop()

    return None


def _phase2_solve(cp, ud, sp, max_depth, last_face):
    if max_depth < 0:
        return None
    if max_depth > _PHASE2_MAX:
        max_depth = _PHASE2_MAX

    h1 = _P2_CP_SLICE[cp * _NPERM4 + sp]
    h2 = _P2_UD_SLICE[ud * _NPERM4 + sp]
    h = h1 if h1 > h2 else h2
    if h > max_depth:
        return None

    for d in range(h, max_depth + 1):
        r = _phase2_dfs(cp, ud, sp, d, last_face, True, [])
        if r is not None:
            return r
    return None


def _phase1_dfs(tw, fl, sl, depth, max_depth, total_depth, last_face, path, start_cube):
    h1 = _P1_TWIST_SLICE[tw * _NSLICE + sl]
    if h1 > max_depth - depth:
        return None
    h2 = _P1_FLIP_SLICE[fl * _NSLICE + sl]
    if h2 > max_depth - depth:
        return None

    if tw == 0 and fl == 0 and sl == _SLICE_GOAL:
        rem2 = total_depth - depth
        if rem2 >= 0:
            g = _apply_path(start_cube, path)
            cp = _get_cp(g)
            ud = _get_ud_edges(g)
            sp = _get_slice_perm(g)
            p2 = _phase2_solve(cp, ud, sp, rem2, last_face)
            if p2 is not None:
                return list(path) + p2

    if depth == max_depth:
        return None

    btw = tw * 18
    bfl = fl * 18
    bsl = sl * 18

    for mid in _P1_MOVE_IDS:
        face = mid // 3
        if _skip_phase1(face, last_face):
            continue

        ntw = _P1_TWIST_MOVE[btw + mid]
        nfl = _P1_FLIP_MOVE[bfl + mid]
        nsl = _P1_SLICE_MOVE[bsl + mid]

        path.append(mid)
        r = _phase1_dfs(ntw, nfl, nsl, depth + 1, max_depth, total_depth, face, path, start_cube)
        if r is not None:
            return r
        path.pop()

    return None


def _simplify_once(moves):
    out = []
    i = 0
    n = len(moves)
    while i < n:
        axis = _AXIS_OF_FACE[moves[i] // 3]
        powers = {f: 0 for f in _AXIS_FACES[axis]}
        j = i
        while j < n and _AXIS_OF_FACE[moves[j] // 3] == axis:
            m = moves[j]
            f = m // 3
            powers[f] = (powers[f] + _POWER_OF[m % 3]) & 3
            j += 1
        for f in _AXIS_FACES[axis]:
            p = powers[f]
            if p:
                out.append(3 * f + _ID_OF_POWER[p])
        i = j
    return out


def _simplify_moves(moves):
    prev = list(moves)
    while True:
        cur = _simplify_once(prev)
        if cur == prev:
            return cur
        prev = cur


def _search(cube):
    if _is_solved(cube):
        return []

    tw = _get_twist(cube)
    fl = _get_flip(cube)
    sl = _get_slice_combo(cube)

    h1 = _P1_TWIST_SLICE[tw * _NSLICE + sl]
    h2 = _P1_FLIP_SLICE[fl * _NSLICE + sl]
    lower = h1 if h1 > h2 else h2

    for total in range(lower, _TOTAL_MAX + 1):
        max_p1 = total if total < _PHASE1_MAX else _PHASE1_MAX
        r = _phase1_dfs(tw, fl, sl, 0, max_p1, total, -1, [], cube)
        if r is not None:
            return _simplify_moves(r)

    return None


def _moves_to_string(moves):
    return " ".join(_MOVE_NAMES[m] for m in moves)


def _basic_validate(facelet):
    if not isinstance(facelet, str):
        raise ValueError("facelet must be a string")
    if len(facelet) != 54:
        raise ValueError("facelet string must have length 54")
    if any(c not in _FACE_SET for c in facelet):
        raise ValueError("invalid facelet character")
    centers = (4, 13, 22, 31, 40, 49)
    for i, ch in zip(centers, _FACE_NAMES):
        if facelet[i] != ch:
            raise ValueError("invalid center facelet")
    for ch in _FACE_NAMES:
        if facelet.count(ch) != 9:
            raise ValueError("invalid facelet counts")


def _parity(p):
    r = 0
    n = len(p)
    for i in range(n - 1):
        pi = p[i]
        for j in range(i + 1, n):
            if pi > p[j]:
                r ^= 1
    return r


def _validate_cubie(c):
    if sorted(c.cp) != _SOLVED_CP:
        raise ValueError("invalid corners")
    if sorted(c.ep) != _SOLVED_EP:
        raise ValueError("invalid edges")
    if sum(c.co) % 3:
        raise ValueError("invalid corner orientation")
    if sum(c.eo) & 1:
        raise ValueError("invalid edge orientation")
    if _parity(c.cp) != _parity(c.ep):
        raise ValueError("invalid permutation parity")


def solve(facelet: str) -> str:
    _basic_validate(facelet)
    if facelet == _SOLVED:
        return ""

    cube = _cubie_from_facelets(facelet)
    _validate_cubie(cube)

    _init()
    solution = _search(cube)
    if solution is None:
        raise ValueError("no solution found")
    if not _is_solved(_apply_path(cube, solution)):
        raise RuntimeError("internal solver error")

    return _moves_to_string(solution)