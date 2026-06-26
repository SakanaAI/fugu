import time

# ---- index tables for facelet <-> cubie conversion ----
cornerFacelet = [
    [8, 9, 20], [6, 18, 38], [0, 36, 47], [2, 45, 11],
    [29, 26, 15], [27, 44, 24], [33, 53, 42], [35, 17, 51]]
cornerColor = [
    ['U', 'R', 'F'], ['U', 'F', 'L'], ['U', 'L', 'B'], ['U', 'B', 'R'],
    ['D', 'F', 'R'], ['D', 'L', 'F'], ['D', 'B', 'L'], ['D', 'R', 'B']]
edgeFacelet = [
    [5, 10], [7, 19], [3, 37], [1, 46],
    [32, 16], [28, 25], [31, 43], [34, 52],
    [23, 14], [21, 41], [50, 39], [48, 12]]
edgeColor = [
    ['U', 'R'], ['U', 'F'], ['U', 'L'], ['U', 'B'],
    ['D', 'R'], ['D', 'F'], ['D', 'L'], ['D', 'B'],
    ['F', 'R'], ['F', 'L'], ['B', 'L'], ['B', 'R']]


def to_cubie(s):
    cp = [0] * 8
    co = [0] * 8
    ep = [0] * 12
    eo = [0] * 12
    for i in range(8):
        ori = 0
        for o in range(3):
            c = s[cornerFacelet[i][o]]
            if c == 'U' or c == 'D':
                ori = o
                break
        col1 = s[cornerFacelet[i][(ori + 1) % 3]]
        col2 = s[cornerFacelet[i][(ori + 2) % 3]]
        for j in range(8):
            if cornerColor[j][1] == col1 and cornerColor[j][2] == col2:
                cp[i] = j
                co[i] = ori
                break
    for i in range(12):
        c0 = s[edgeFacelet[i][0]]
        c1 = s[edgeFacelet[i][1]]
        for j in range(12):
            if c0 == edgeColor[j][0] and c1 == edgeColor[j][1]:
                ep[i] = j
                eo[i] = 0
                break
            if c0 == edgeColor[j][1] and c1 == edgeColor[j][0]:
                ep[i] = j
                eo[i] = 1
                break
    return cp, co, ep, eo


# ---- basic moves (cubie level) ----
moveU = ([3, 0, 1, 2, 4, 5, 6, 7], [0] * 8,
         [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], [0] * 12)
moveR = ([4, 1, 2, 0, 7, 5, 6, 3], [2, 0, 0, 1, 1, 0, 0, 2],
         [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0], [0] * 12)
moveF = ([1, 5, 2, 3, 0, 4, 6, 7], [1, 2, 0, 0, 2, 1, 0, 0],
         [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
         [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0])
moveD = ([0, 1, 2, 3, 5, 6, 7, 4], [0] * 8,
         [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11], [0] * 12)
moveL = ([0, 2, 6, 3, 4, 1, 5, 7], [0, 1, 2, 0, 0, 2, 1, 0],
         [0, 1, 6, 3, 4, 5, 10, 7, 8, 2, 9, 11], [0] * 12)
moveB = ([0, 1, 3, 7, 4, 5, 2, 6], [0, 0, 1, 2, 0, 0, 2, 1],
         [0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 3, 10],
         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1])

basics = [moveU, moveR, moveF, moveD, moveL, moveB]


def mult(a, b):
    cp = [a[0][b[0][i]] for i in range(8)]
    co = [(a[1][b[0][i]] + b[1][i]) % 3 for i in range(8)]
    ep = [a[2][b[2][i]] for i in range(12)]
    eo = [(a[3][b[2][i]] + b[3][i]) % 2 for i in range(12)]
    return (cp, co, ep, eo)


moveCube = [None] * 18
for f in range(6):
    m = basics[f]
    cur = m
    moveCube[f * 3 + 0] = cur
    cur = mult(cur, m)
    moveCube[f * 3 + 1] = cur
    cur = mult(cur, m)
    moveCube[f * 3 + 2] = cur

phase2_moves = [0, 1, 2, 9, 10, 11, 4, 13, 7, 16]

faces = "URFDLB"


def move_name(m):
    return faces[m // 3] + ("", "2", "'")[m % 3]


def allowed(prev, cur):
    if prev == -1:
        return True
    pf = prev // 3
    cf = cur // 3
    if pf == cf:
        return False
    if pf % 3 == cf % 3 and pf > cf:
        return False
    return True


# ---- coordinate helpers ----
FACT = [1, 1, 2, 6, 24, 120, 720, 5040, 40320]


def perm_rank(p):
    n = len(p)
    r = 0
    for i in range(n):
        c = 0
        pi = p[i]
        for j in range(i + 1, n):
            if p[j] < pi:
                c += 1
        r += c * FACT[n - 1 - i]
    return r


def perm_unrank(r, n):
    elems = list(range(n))
    p = []
    for i in range(n):
        fct = FACT[n - 1 - i]
        idx = r // fct
        r -= idx * fct
        p.append(elems[idx])
        del elems[idx]
    return p


def co_to_twist(co):
    t = 0
    for i in range(7):
        t = t * 3 + co[i]
    return t


def twist_to_co(t):
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = t % 3
        t //= 3
        s += co[i]
    co[7] = (-s) % 3
    return co


def eo_to_flip(eo):
    f = 0
    for i in range(11):
        f = f * 2 + eo[i]
    return f


def flip_to_eo(f):
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = f & 1
        f >>= 1
        s += eo[i]
    eo[11] = s & 1
    return eo


# binomial
C = [[0] * 13 for _ in range(13)]
for n in range(13):
    C[n][0] = 1
    for k in range(1, n + 1):
        C[n][k] = C[n - 1][k - 1] + C[n - 1][k]


def slice_rank(ep):
    r = 0
    k = 1
    for i in range(12):
        if ep[i] >= 8:
            r += C[i][k]
            k += 1
    return r


def slice_unrank(idx):
    pos = []
    x = idx
    for k in range(4, 0, -1):
        c = k - 1
        while C[c + 1][k] <= x:
            c += 1
        x -= C[c][k]
        pos.append(c)
    pos.sort()
    ep = [-1] * 12
    for j, p in enumerate(pos):
        ep[p] = 8 + j
    nxt = 0
    for i in range(12):
        if ep[i] == -1:
            ep[i] = nxt
            nxt += 1
    return ep


SOLVED_SLICE = slice_rank(list(range(12)))

NT = 2187
NF = 2048
NS = 495
NCP = 40320
NEP = 40320
NSP = 24

# ---- move tables (flat, stride 18) ----
twistMove = [0] * (NT * 18)
for t in range(NT):
    co = twist_to_co(t)
    st = ([0, 1, 2, 3, 4, 5, 6, 7], co, list(range(12)), [0] * 12)
    base = t * 18
    for mm in range(18):
        twistMove[base + mm] = co_to_twist(mult(st, moveCube[mm])[1])

flipMove = [0] * (NF * 18)
for fl in range(NF):
    eo = flip_to_eo(fl)
    st = (list(range(8)), [0] * 8, list(range(12)), eo)
    base = fl * 18
    for mm in range(18):
        flipMove[base + mm] = eo_to_flip(mult(st, moveCube[mm])[3])

sliceMove = [0] * (NS * 18)
for sx in range(NS):
    ep = slice_unrank(sx)
    st = (list(range(8)), [0] * 8, ep, [0] * 12)
    base = sx * 18
    for mm in range(18):
        sliceMove[base + mm] = slice_rank(mult(st, moveCube[mm])[2])

PERMS8 = [perm_unrank(r, 8) for r in range(NCP)]

cpermMove = [0] * (NCP * 18)
for r in range(NCP):
    cp = PERMS8[r]
    base = r * 18
    for mm in phase2_moves:
        Pm = moveCube[mm][0]
        cpermMove[base + mm] = perm_rank([cp[Pm[i]] for i in range(8)])

epermMove = [0] * (NEP * 18)
for r in range(NEP):
    p = PERMS8[r]
    base = r * 18
    for mm in phase2_moves:
        Em = moveCube[mm][2]
        epermMove[base + mm] = perm_rank([p[Em[i]] for i in range(8)])

spermMove = [0] * (NSP * 18)
for r in range(NSP):
    p = perm_unrank(r, 4)
    base = r * 18
    for mm in phase2_moves:
        Em = moveCube[mm][2]
        sp = [p[Em[8 + i] - 8] for i in range(4)]
        spermMove[base + mm] = perm_rank(sp)


# ---- pruning tables ----
def build_pruning(sa, sb, mta, mtb, moves, ga, gb):
    N = sa * sb
    dist = bytearray([255]) * N
    start = ga * sb + gb
    dist[start] = 0
    frontier = [start]
    d = 0
    moves = list(moves)
    while frontier:
        nd = d + 1
        nxt = []
        for idx in frontier:
            a = idx // sb
            b = idx - a * sb
            ba = a * 18
            bb = b * 18
            for mm in moves:
                ni = mta[ba + mm] * sb + mtb[bb + mm]
                if dist[ni] == 255:
                    dist[ni] = nd
                    nxt.append(ni)
        frontier = nxt
        d = nd
    return dist


pdt_ts = build_pruning(NT, NS, twistMove, sliceMove, range(18), 0, SOLVED_SLICE)
pdt_fs = build_pruning(NF, NS, flipMove, sliceMove, range(18), 0, SOLVED_SLICE)
pdt_cs = build_pruning(NCP, NSP, cpermMove, spermMove, phase2_moves, 0, 0)
pdt_es = build_pruning(NEP, NSP, epermMove, spermMove, phase2_moves, 0, 0)


def simplify(moves):
    res = []
    for m in moves:
        f = m // 3
        t = m % 3 + 1
        if res and res[-1][0] == f:
            nt = (res[-1][1] + t) % 4
            res.pop()
            if nt != 0:
                res.append((f, nt))
        else:
            res.append((f, t))
    out = []
    for f, t in res:
        out.append(f * 3 + (t - 1))
    return out


def solve(facelet):
    s = facelet.strip()
    cp, co, ep, eo = to_cubie(s)
    init = (cp, co, ep, eo)
    twist0 = co_to_twist(co)
    flip0 = eo_to_flip(eo)
    slc0 = slice_rank(ep)

    if (twist0 == 0 and flip0 == 0 and slc0 == SOLVED_SLICE
            and cp == list(range(8)) and ep == list(range(12))):
        return ""

    start = time.time()
    deadline = start + 0.5

    best = [None]
    best_len = [999]
    stop = [False]
    counter = [0]

    def solve_phase2(cube, maxlen):
        cpc = perm_rank(cube[0])
        epc = perm_rank(cube[2][0:8])
        spc = perm_rank([cube[2][8 + i] - 8 for i in range(4)])
        if maxlen > 18:
            maxlen = 18

        def dfs(cpv, epv, spv, depth, last):
            if depth == 0:
                if cpv == 0 and epv == 0 and spv == 0:
                    return []
                return None
            if pdt_cs[cpv * 24 + spv] > depth:
                return None
            if pdt_es[epv * 24 + spv] > depth:
                return None
            for mm in phase2_moves:
                if not allowed(last, mm):
                    continue
                r = dfs(cpermMove[cpv * 18 + mm], epermMove[epv * 18 + mm],
                        spermMove[spv * 18 + mm], depth - 1, mm)
                if r is not None:
                    return [mm] + r
            return None

        for d in range(0, maxlen + 1):
            r = dfs(cpc, epc, spc, d, -1)
            if r is not None:
                return r
        return None

    def phase1(tw, fl, sl, depth, last, moves):
        if stop[0]:
            return
        counter[0] += 1
        if (counter[0] & 1023) == 0 and best[0] is not None and time.time() > deadline:
            stop[0] = True
            return
        if depth == 0:
            if tw == 0 and fl == 0 and sl == SOLVED_SLICE:
                cube = init
                for mm in moves:
                    cube = mult(cube, moveCube[mm])
                maxp2 = best_len[0] - len(moves) - 1
                if maxp2 < 0:
                    return
                p2 = solve_phase2(cube, maxp2)
                if p2 is not None:
                    total = simplify(moves + p2)
                    if len(total) < best_len[0]:
                        best_len[0] = len(total)
                        best[0] = total
            return
        if pdt_ts[tw * NS + sl] > depth:
            return
        if pdt_fs[fl * NS + sl] > depth:
            return
        for mm in range(18):
            if not allowed(last, mm):
                continue
            phase1(twistMove[tw * 18 + mm], flipMove[fl * 18 + mm],
                   sliceMove[sl * 18 + mm], depth - 1, mm, moves + [mm])
            if stop[0]:
                return

    for d1 in range(0, 13):
        if best[0] is not None and d1 >= best_len[0]:
            break
        if stop[0]:
            break
        phase1(twist0, flip0, slc0, d1, -1, [])
        if stop[0]:
            break

    if best[0] is None:
        return ""
    return " ".join(move_name(m) for m in best[0])