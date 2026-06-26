import time as _time
from itertools import permutations as _perms, combinations as _combs

# ---------------------------------------------------------------------------
# Two-phase (Kociemba) Rubik's cube solver, pure standard library.
# Faces: U=0 R=1 F=2 D=3 L=4 B=5
# Corners: URF UFL ULB UBR DFR DLF DBL DRB
# Edges:   UR UF UL UB DR DF DL DB FR FL BL BR
# ---------------------------------------------------------------------------

_CPm = ((3, 0, 1, 2, 4, 5, 6, 7),
        (4, 1, 2, 0, 7, 5, 6, 3),
        (1, 5, 2, 3, 0, 4, 6, 7),
        (0, 1, 2, 3, 5, 6, 7, 4),
        (0, 2, 6, 3, 4, 1, 5, 7),
        (0, 1, 3, 7, 4, 5, 2, 6))
_COm = ((0, 0, 0, 0, 0, 0, 0, 0),
        (2, 0, 0, 1, 1, 0, 0, 2),
        (1, 2, 0, 0, 2, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 2, 0, 0, 2, 1, 0),
        (0, 0, 1, 2, 0, 0, 2, 1))
_EPm = ((3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        (8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0),
        (0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11),
        (0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11),
        (0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11),
        (0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7))
_EOm = ((0,) * 12,
        (0,) * 12,
        (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
        (0,) * 12,
        (0,) * 12,
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1))

_CORNER_FACELET = ((8, 9, 20), (6, 18, 38), (0, 36, 47), (2, 45, 11),
                   (29, 26, 15), (27, 44, 24), (33, 53, 42), (35, 17, 51))
_CORNER_COLOR = ((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1),
                 (3, 2, 1), (3, 4, 2), (3, 5, 4), (3, 1, 5))
_EDGE_FACELET = ((5, 10), (7, 19), (3, 37), (1, 46), (32, 16), (28, 25),
                 (30, 43), (34, 52), (23, 12), (21, 41), (50, 39), (48, 14))
_EDGE_COLOR = ((0, 1), (0, 2), (0, 4), (0, 5), (3, 1), (3, 2),
               (3, 4), (3, 5), (2, 1), (2, 4), (5, 4), (5, 1))

_NAMES = ("U", "U2", "U'", "R", "R2", "R'", "F", "F2", "F'",
          "D", "D2", "D'", "L", "L2", "L'", "B", "B2", "B'")
_ID8 = tuple(range(8))
_ID12 = tuple(range(12))

_BUDGET = 4.0
_STOP_AT = 20


class _Stop(Exception):
    pass


def _par(p):
    inv = 0
    n = len(p)
    for i in range(n):
        pi = p[i]
        for j in range(i + 1, n):
            if pi > p[j]:
                inv += 1
    return inv & 1


def _build_prun(mt1, mt2, n2, nm, start):
    n1 = len(mt1) // nm
    size = n1 * n2
    mt1p = [v * n2 for v in mt1]
    dist = bytearray(b'\xff') * size
    dist[start] = 0
    frontier = [start]
    d = 0
    while frontier:
        nxt = []
        app = nxt.append
        nd = d + 1
        for s in frontier:
            a, b = divmod(s, n2)
            a *= nm
            b *= nm
            for v, w in zip(mt1p[a:a + nm], mt2[b:b + nm]):
                t = v + w
                if dist[t] == 255:
                    dist[t] = nd
                    app(t)
        frontier = nxt
        d = nd
    return dist


def _init():
    global _MCP, _MEP, _TWM, _FLM, _SLM, _CRANK, _SLICE0
    global _RANK8, _RANK4, _P2M, _P2F, _CP2, _EP2, _SP2
    global _PR1F, _PR1T, _PR2C, _PR2E

    # net cubie permutations for all 18 moves
    _MCP = []
    _MEP = []
    for f in range(6):
        cpf = _CPm[f]
        epf = _EPm[f]
        c = tuple(range(8))
        e = tuple(range(12))
        for _p in range(3):
            c = tuple(c[cpf[i]] for i in range(8))
            e = tuple(e[epf[i]] for i in range(12))
            _MCP.append(c)
            _MEP.append(e)

    # corner-orientation (twist) move table
    TW = [0] * (2187 * 18)
    for t in range(2187):
        co = [0] * 8
        x = t
        for i in range(6, -1, -1):
            co[i] = x % 3
            x //= 3
        co[7] = (-sum(co[:7])) % 3
        b = t * 18
        for f in range(6):
            cpf = _CPm[f]
            com = _COm[f]
            c = co
            for p in range(3):
                c = [(c[cpf[i]] + com[i]) % 3 for i in range(8)]
                y = 0
                for i in range(7):
                    y = 3 * y + c[i]
                TW[b + f * 3 + p] = y
    _TWM = TW

    # edge-orientation (flip) move table
    FL = [0] * (2048 * 18)
    for t in range(2048):
        eo = [0] * 12
        x = t
        for i in range(10, -1, -1):
            eo[i] = x & 1
            x >>= 1
        eo[11] = sum(eo[:11]) & 1
        b = t * 18
        for f in range(6):
            epf = _EPm[f]
            eom = _EOm[f]
            e = eo
            for p in range(3):
                e = [(e[epf[i]] + eom[i]) & 1 for i in range(12)]
                y = 0
                for i in range(11):
                    y = 2 * y + e[i]
                FL[b + f * 3 + p] = y
    _FLM = FL

    # UD-slice edge location move table
    COMB = list(_combs(range(12), 4))
    _CRANK = {c: i for i, c in enumerate(COMB)}
    _SLICE0 = _CRANK[(8, 9, 10, 11)]
    SL = [0] * (495 * 18)
    for idx, cmb in enumerate(COMB):
        occ0 = [0] * 12
        for j in cmb:
            occ0[j] = 1
        b = idx * 18
        for f in range(6):
            epf = _EPm[f]
            o = occ0
            for p in range(3):
                o = [o[epf[i]] for i in range(12)]
                SL[b + f * 3 + p] = _CRANK[tuple(j for j in range(12) if o[j])]
    _SLM = SL

    # phase-2 permutation coordinates
    P8 = list(_perms(range(8)))
    _RANK8 = {p: i for i, p in enumerate(P8)}
    P4 = list(_perms(range(4)))
    _RANK4 = {p: i for i, p in enumerate(P4)}

    _P2M = (0, 1, 2, 9, 10, 11, 4, 13, 7, 16)   # U U2 U' D D2 D' R2 L2 F2 B2
    _P2F = (0, 0, 0, 3, 3, 3, 1, 4, 2, 5)
    sc_list = [_MCP[m] for m in _P2M]
    se_list = [_MEP[m][:8] for m in _P2M]
    ss_list = [tuple(_MEP[m][i] - 8 for i in (8, 9, 10, 11)) for m in _P2M]

    def ptable(sigmas):
        cols = []
        r = _RANK8
        for s in sigmas:
            a0, a1, a2, a3, a4, a5, a6, a7 = s
            cols.append([r[(p[a0], p[a1], p[a2], p[a3],
                            p[a4], p[a5], p[a6], p[a7])] for p in P8])
        flat = [0] * (40320 * 10)
        i = 0
        for row in zip(*cols):
            flat[i:i + 10] = row
            i += 10
        return flat

    _CP2 = ptable(sc_list)
    _EP2 = ptable(se_list)

    SP = [0] * (24 * 10)
    for i, p in enumerate(P4):
        b = i * 10
        for k in range(10):
            s = ss_list[k]
            SP[b + k] = _RANK4[(p[s[0]], p[s[1]], p[s[2]], p[s[3]])]
    _SP2 = SP

    # pruning tables
    _PR1F = _build_prun(_FLM, _SLM, 495, 18, _SLICE0)
    _PR1T = _build_prun(_TWM, _SLM, 495, 18, _SLICE0)
    _PR2C = _build_prun(_CP2, _SP2, 24, 10, 0)
    _PR2E = _build_prun(_EP2, _SP2, 24, 10, 0)


_init()


def solve(facelet: str) -> str:
    fstr = facelet.strip()
    if len(fstr) != 54:
        raise ValueError("facelet string must contain 54 characters")
    cmap = {fstr[4]: 0, fstr[13]: 1, fstr[22]: 2,
            fstr[31]: 3, fstr[40]: 4, fstr[49]: 5}
    fc = [cmap[ch] for ch in fstr]

    cp = [0] * 8
    co = [0] * 8
    ep = [0] * 12
    eo = [0] * 12
    for i in range(8):
        fs = _CORNER_FACELET[i]
        ori = 0
        for o in range(3):
            if fc[fs[o]] == 0 or fc[fs[o]] == 3:
                ori = o
                break
        c1 = fc[fs[(ori + 1) % 3]]
        c2 = fc[fs[(ori + 2) % 3]]
        for j in range(8):
            col = _CORNER_COLOR[j]
            if c1 == col[1] and c2 == col[2]:
                cp[i] = j
                co[i] = ori
                break
    for i in range(12):
        a = fc[_EDGE_FACELET[i][0]]
        b = fc[_EDGE_FACELET[i][1]]
        for j in range(12):
            col = _EDGE_COLOR[j]
            if a == col[0] and b == col[1]:
                ep[i] = j
                eo[i] = 0
                break
            if a == col[1] and b == col[0]:
                ep[i] = j
                eo[i] = 1
                break
    cp0 = tuple(cp)
    ep0 = tuple(ep)

    if (sorted(cp0) != list(range(8)) or sorted(ep0) != list(range(12))
            or sum(co) % 3 or sum(eo) % 2 or _par(cp0) != _par(ep0)):
        raise ValueError("invalid cube state")

    tw0 = 0
    for i in range(7):
        tw0 = 3 * tw0 + co[i]
    fl0 = 0
    for i in range(11):
        fl0 = 2 * fl0 + eo[i]
    sl0 = _CRANK[tuple(i for i in range(12) if ep0[i] >= 8)]

    if tw0 == 0 and fl0 == 0 and cp0 == _ID8 and ep0 == _ID12:
        return ""

    deadline = _time.perf_counter() + _BUDGET
    hard = deadline + 3.0 * _BUDGET
    best = None
    best_len = 31
    moves1 = [0] * 26
    moves2 = [0] * 26
    ncnt = 0
    fail = {}

    def p2(c, e, s, depth, lf, ply,
           CP2=_CP2, EP2=_EP2, SP2=_SP2, PC=_PR2C, PE=_PR2E,
           P2F=_P2F, P2M=_P2M, clock=_time.perf_counter):
        nonlocal ncnt
        if depth == 0:
            return c == 0 and e == 0 and s == 0
        if PC[c * 24 + s] > depth or PE[e * 24 + s] > depth:
            return False
        ncnt += 1
        if not ncnt & 4095:
            t = clock()
            if t > deadline and (best is not None or t > hard):
                raise _Stop
        cb = c * 10
        eb = e * 10
        sb = s * 10
        dn = depth - 1
        pn = ply + 1
        for k in range(10):
            fk = P2F[k]
            if fk == lf or fk == lf + 3:
                continue
            moves2[ply] = P2M[k]
            if p2(CP2[cb + k], EP2[eb + k], SP2[sb + k], dn, fk, pn):
                return True
        return False

    def g1(d1):
        nonlocal best, best_len
        if d1:
            m = moves1[d1 - 1]
            lf = m // 3
            if lf == 0 or lf == 3 or m % 3 == 1:
                return          # avoid duplicates: last move must be RLFB quarter
        else:
            lf = 6
        lim = best_len - d1 - 1
        if lim > 18:
            lim = 18
        if lim < 0:
            return
        cpx = cp0
        epx = ep0
        for i in range(d1):
            mi = moves1[i]
            sc = _MCP[mi]
            se = _MEP[mi]
            cpx = (cpx[sc[0]], cpx[sc[1]], cpx[sc[2]], cpx[sc[3]],
                   cpx[sc[4]], cpx[sc[5]], cpx[sc[6]], cpx[sc[7]])
            epx = (epx[se[0]], epx[se[1]], epx[se[2]], epx[se[3]],
                   epx[se[4]], epx[se[5]], epx[se[6]], epx[se[7]],
                   epx[se[8]], epx[se[9]], epx[se[10]], epx[se[11]])
        c = _RANK8[cpx]
        e = _RANK8[epx[:8]]
        s = _RANK4[(epx[8] - 8, epx[9] - 8, epx[10] - 8, epx[11] - 8)]
        h = _PR2C[c * 24 + s]
        h2 = _PR2E[e * 24 + s]
        if h2 > h:
            h = h2
        if h > lim:
            return
        key = (c, e, s, lf)
        prev = fail.get(key, -1)
        if lim <= prev:
            return
        start = h if h > prev + 1 else prev + 1
        for d2 in range(start, lim + 1):
            if p2(c, e, s, d2, lf, 0):
                fail[key] = d2 - 1
                best = moves1[:d1] + moves2[:d2]
                best_len = d1 + d2
                if best_len <= _STOP_AT:
                    raise _Stop
                return
        fail[key] = lim

    def p1(tw, fl, sl, depth, lf, ply,
           TW=_TWM, FL=_FLM, SL=_SLM, PF=_PR1F, PT=_PR1T,
           S0=_SLICE0, clock=_time.perf_counter):
        nonlocal ncnt
        if depth == 0:
            if sl == S0 and fl == 0 and tw == 0:
                g1(ply)
            return
        if PF[fl * 495 + sl] > depth or PT[tw * 495 + sl] > depth:
            return
        ncnt += 1
        if not ncnt & 4095:
            t = clock()
            if t > deadline and (best is not None or t > hard):
                raise _Stop
        tb = tw * 18
        fb = fl * 18
        sb = sl * 18
        dn = depth - 1
        pn = ply + 1
        for f in range(6):
            if f == lf or f == lf + 3:
                continue
            m = f * 3
            moves1[ply] = m
            p1(TW[tb + m], FL[fb + m], SL[sb + m], dn, f, pn)
            m += 1
            moves1[ply] = m
            p1(TW[tb + m], FL[fb + m], SL[sb + m], dn, f, pn)
            m += 1
            moves1[ply] = m
            p1(TW[tb + m], FL[fb + m], SL[sb + m], dn, f, pn)

    h1 = _PR1F[fl0 * 495 + sl0]
    t1 = _PR1T[tw0 * 495 + sl0]
    if t1 > h1:
        h1 = t1
    try:
        d1 = h1
        while d1 < best_len and d1 <= 20:
            p1(tw0, fl0, sl0, d1, 6, 0)
            d1 += 1
    except _Stop:
        pass

    if best is None:
        raise ValueError("no solution found (invalid cube state?)")
    return " ".join(_NAMES[m] for m in best)