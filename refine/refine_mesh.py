#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_mesh.py
==============
تحسين سطح (Mesh) ناتج عن بواسون: تنعيم يحافظ على الحواف الحقيقية، مع خيارات
لتنظيم المناطق المستوية (جدران، فتحات) وتشديد التنعيم على المناطق التي "أكملتها"
خوارزمية بواسون (الملوّنة بلون التمييز)، وخريطة انحراف توثّق ما تغيّر فعليًا.

الفكرة (مبنية على أفكار تنقية الـ mesh بحفظ الميزات، مثل Sun et al. 2007
و Zheng et al. 2011)
------------------------------------------------------------------------------
1) تنعيم اتجاهات المثلثات (normals) بتثقيل ثنائي (Bilateral): كل مثلث يأخذ
   متوسط جيرانه المشابهين له بالاتجاه *فقط*، فلا يُخلَط وجهان بينهما حافة حادة،
   بينما الضجيج (اختلافات صغيرة بالاتجاه) يُلغى. هذا هو "التثقيل".
2) تحريك الرؤوس لتطابق الاتجاهات المنعّمة، مع سقف أقصى لإزاحة أي رأس عن موضعه
   الأصلي (--max-disp) — ضمانة هندسية: لا رأس يتحرك أكثر من التسامح المحدد.
3) تشديد التنعيم على المناطق التي أكملتها بواسون (لا بيانات قياس حقيقية فيها)،
   بتمييزها من لون التمييز نفسه الذي وضعه سكربت بواسون.
4) اختياري (--planes): كشف المناطق المستوية فعلًا (ضمن تسامح) وسحب رؤوسها إلى
   المستوى المُلاءَم، وعند التقاء مستويين/ثلاثة تُشَدّ الرؤوس إلى خط/نقطة التقاطع
   فتظهر الزوايا حادة ونظيفة. المناطق غير المستوية (مقرنصات، أعمدة، زخارف) لا تُمَسّ.
5) خريطة انحراف (ملف PLY ملوّن) + تقرير JSON: كم تحرك كل رأس عن الأصل. توثيق
   مهم في HBIM كي لا نفقد معلومة الـ as-built.

لا يغيّر عدد الرؤوس ولا ربط المثلثات (باستثناء ملف التبسيط الاختياري المنفصل)،
فألوان الرؤوس الأصلية تبقى كما هي.

المتطلبات
---------
    pip install open3d numpy
    pip install scipy psutil   # اختياريتان: scipy لتسريع إعادة تلوين التبسيط، psutil لقراءة الرام على Windows/macOS

أمثلة
-----
    # الافتراضي: مستوى medium
    python3 refine_mesh.py -i mesh.ply

    # ثلاث درجات دفعة واحدة (light, medium, strong) للمقارنة البصرية
    python3 refine_mesh.py -i mesh.ply --level light,medium,strong

    # مع تنظيم المستويات (جدران/فتحات) وتسامح 2 سم (إن كانت الوحدة متر)
    python3 refine_mesh.py -i mesh.ply --planes --plane-tol 0.02

    # حفاظ أكبر على التفاصيل الدقيقة (زخارف): زاوية ميزة أصغر وسقف إزاحة صغير
    python3 refine_mesh.py -i mesh.ply --feature-angle 15 --max-disp 0.004

    # ثم نسخة مبسّطة 200 ألف مثلث للتصدير (بعد التنعيم لا قبله)
    python3 refine_mesh.py -i mesh.ply --simplify-to 200000
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import open3d as o3d

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def log(msg=""):
    print(msg, flush=True)


def human_time(s):
    if s < 60:
        return f"{s:.1f} ثانية"
    if s < 3600:
        return f"{s / 60:.1f} دقيقة"
    return f"{s / 3600:.2f} ساعة"


# =============================================================================
# معلومات الذاكرة (psutil إن توفرت، وإلا /proc على لينكس)
# =============================================================================

def get_mem_info():
    if _HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            return {"MemTotal": vm.total / 1024.0 ** 3, "MemAvailable": vm.available / 1024.0 ** 3}
        except Exception:  # noqa: BLE001
            pass
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(rest.strip().split()[0]) / 1024.0 ** 2
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return info


# =============================================================================
# طوبولوجيا السطح (تُحسب مرة واحدة وتُعاد استخدامها)
# =============================================================================

def build_topology(F, n_vertices):
    """أزواج المثلثات المتجاورة عبر ضلع مشترك (موجَّهة بالاتجاهين) + الأضلاع الفريدة."""
    nF = len(F)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    face_id = np.tile(np.arange(nF, dtype=np.int64), 3)
    Es = np.sort(E, axis=1)
    key = Es[:, 0].astype(np.int64) * n_vertices + Es[:, 1]
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    fid_s = face_id[order]
    same = key_s[1:] == key_s[:-1]
    a = fid_s[:-1][same]
    b = fid_s[1:][same]
    keep = a != b
    a, b = a[keep], b[keep]
    Pi = np.concatenate([a, b])
    Pj = np.concatenate([b, a])
    _, first = np.unique(key, return_index=True)
    edges_u = Es[first]
    return {"Pi": Pi, "Pj": Pj, "edges": edges_u}


def face_geometry(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(area2, 1e-30)[:, None]
    n[area2 < 1e-30] = (0.0, 0.0, 1.0)
    return n, 0.5 * area2, (a + b + c) / 3.0


def median_edge_length(V, edges):
    d = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    return float(np.median(d)) if len(d) else 0.0


def vertex_neighbor_mean(vals, edges, deg):
    """متوسط قيم الجيران (umbrella operator). vals: (nV,) أو (nV,3)."""
    nV = len(deg)
    if vals.ndim == 1:
        s = (np.bincount(edges[:, 0], weights=vals[edges[:, 1]], minlength=nV)
             + np.bincount(edges[:, 1], weights=vals[edges[:, 0]], minlength=nV))
        return s / np.maximum(deg, 1)
    out = np.empty_like(vals)
    for k in range(vals.shape[1]):
        s = (np.bincount(edges[:, 0], weights=vals[edges[:, 1], k], minlength=nV)
             + np.bincount(edges[:, 1], weights=vals[edges[:, 0], k], minlength=nV))
        out[:, k] = s / np.maximum(deg, 1)
    # رؤوس بلا جيران تبقى كما هي
    iso = deg == 0
    out[iso] = vals[iso]
    return out


# =============================================================================
# 1) تنعيم الاتجاهات بتثقيل ثنائي + 2) تحريك الرؤوس
# =============================================================================

def filter_normals(N, A, Pi, Pj, sigma_r, iters):
    """كل مثلث يأخذ متوسطًا مثقَّلًا بالمساحة وبتشابه الاتجاه لجيرانه.
    وزن الجار = مساحته × exp(-((1 - cosθ)/sigma_r)^2): جار بزاوية كبيرة (حافة) وزنه ≈ 0."""
    nF = len(N)
    cur = N.copy()
    for _ in range(iters):
        dot = np.clip(np.einsum("ij,ij->i", cur[Pi], cur[Pj]), -1.0, 1.0)
        wr = np.exp(-(((1.0 - dot) / sigma_r) ** 2))
        w = A[Pj] * wr
        acc = A[:, None] * cur
        for k in range(3):
            acc[:, k] += np.bincount(Pi, weights=w * cur[Pj, k], minlength=nF)
        nrm = np.maximum(np.linalg.norm(acc, axis=1), 1e-30)
        cur = acc / nrm[:, None]
    return cur


def update_vertices(X, F, Nf, iters, step):
    """تحريك كل رأس ليصبح مستويات المثلثات المحيطة به (باتجاهاتها المنعّمة) أقرب لتمرّ بالرأس."""
    nV = len(X)
    cnt = np.maximum(np.bincount(F.ravel(), minlength=nV).astype(np.float64), 1.0)
    X = X.copy()
    for _ in range(iters):
        C = (X[F[:, 0]] + X[F[:, 1]] + X[F[:, 2]]) / 3.0
        disp = np.zeros_like(X)
        for corner in range(3):
            vi = F[:, corner]
            t = np.einsum("ij,ij->i", Nf, C - X[vi])
            contrib = Nf * t[:, None]
            for k in range(3):
                disp[:, k] += np.bincount(vi, weights=contrib[:, k], minlength=nV)
        X += step * disp / cnt[:, None]
    return X


def clamp_displacement(X, V0, max_disp):
    """سقف لإزاحة كل رأس عن موضعه الأصلي (max_disp قيمة واحدة أو مصفوفة لكل رأس)."""
    d = X - V0
    n = np.linalg.norm(d, axis=1)
    lim = np.broadcast_to(np.asarray(max_disp, dtype=np.float64), n.shape)
    scale = np.minimum(1.0, lim / np.maximum(n, 1e-30))
    return V0 + d * scale[:, None]


def taubin_masked(X, edges, deg, weight, iters, lam=0.5, mu=-0.53):
    """تنعيم Taubin (بلا انكماش تقريبًا) يُطبَّق بوزن لكل رأس (0 = لا تغيير)."""
    w = weight[:, None]
    for _ in range(iters):
        X = X + lam * w * (vertex_neighbor_mean(X, edges, deg) - X)
        X = X + mu * w * (vertex_neighbor_mean(X, edges, deg) - X)
    return X


# =============================================================================
# 4) كشف المناطق المستوية وتنظيمها (اختياري)
# =============================================================================

def _fit_plane(P, w, n_ref):
    mu = (w[:, None] * P).sum(0) / w.sum()
    Q = P - mu
    cov = (Q * w[:, None]).T @ Q / w.sum()
    vals, vecs = np.linalg.eigh(cov)
    n = vecs[:, 0]
    if n @ n_ref < 0:
        n = -n
    rms = math.sqrt(float((w * (Q @ n) ** 2).sum() / w.sum()))
    # العرض المكافئ (بالاتجاه الأصغر ضمن المستوى): شريحة منتظمة بعرض w لها تباين w²/12
    width = math.sqrt(12.0 * max(float(vals[1]), 0.0))
    return n, float(n @ mu), rms, width


def regularize_planes(X, F, topo, angle_tol_deg, plane_tol, min_faces, boundary_w, min_width=0.0, log_fn=log):
    nV, nF = len(X), len(F)
    Nf, Af, Cf = face_geometry(X, F)
    Pi, Pj = topo["Pi"], topo["Pj"]

    order = np.argsort(Pi, kind="stable")
    nbr = Pj[order]
    counts = np.bincount(Pi, minlength=nF)
    indptr = np.zeros(nF + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    dots = np.einsum("ij,ij->i", Nf[Pi], Nf[Pj])
    flat = np.bincount(Pi, weights=1.0 - dots, minlength=nF) / np.maximum(counts, 1)
    cos_tol = math.cos(math.radians(angle_tol_deg))
    seed_ok = flat < (1.0 - cos_tol)
    seeds = np.argsort(flat, kind="stable")
    seeds = seeds[seed_ok[seeds]]

    labels = np.full(nF, -1, dtype=np.int32)
    tried = np.zeros(nF, dtype=bool)
    mark = np.zeros(nF, dtype=bool)

    def gather(frontier):
        starts = indptr[frontier]
        lens = indptr[frontier + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offs = np.repeat(starts - (np.cumsum(lens) - lens), lens)
        return nbr[np.arange(total) + offs]

    def grow(seed, n0, d0):
        mark[seed] = True
        frontier = np.array([seed], dtype=np.int64)
        parts = [frontier]
        while True:
            cand = gather(frontier)
            if cand.size == 0:
                break
            cand = cand[(~mark[cand]) & (labels[cand] < 0)]
            if cand.size == 0:
                break
            cand = np.unique(cand)
            ok = (Nf[cand] @ n0 > cos_tol) & (np.abs(Cf[cand] @ n0 - d0) < plane_tol)
            cand = cand[ok]
            if cand.size == 0:
                break
            mark[cand] = True
            parts.append(cand)
            frontier = cand
        region = np.concatenate(parts)
        mark[region] = False
        return region

    plane_n, plane_d, plane_faces = [], [], []
    for s in seeds:
        if labels[s] >= 0 or tried[s]:
            continue
        n0 = Nf[s]
        d0 = float(n0 @ Cf[s])
        region = grow(s, n0, d0)
        # إعادة ملاءمة المستوى ثم نمو ثانٍ بالمستوى المحسَّن (يُوصل المساحات الطويلة)
        for _round in range(2):
            n_fit, d_fit, _rms, _w = _fit_plane(Cf[region], Af[region], Nf[s])
            region = grow(s, n_fit, d_fit)
        accepted = False
        if region.size >= min_faces:
            n_fit, d_fit, rms, width = _fit_plane(Cf[region], Af[region], Nf[s])
            # شريحة رفيعة جدًا (سنتيمترات) غالبًا جزء من سطح منحنٍ (عمود، قبة) لا جدار حقيقي؛
            # الجدار/الفتحة الحقيقية عريضة بالاتجاهين.
            accepted = width >= min_width
        if accepted:
            labels[region] = len(plane_n)
            plane_n.append(n_fit)
            plane_d.append(d_fit)
            plane_faces.append(int(region.size))
        else:
            tried[region] = True

    L = len(plane_n)
    info = {"n_planes": L, "plane_faces": plane_faces, "n_planar_faces": int((labels >= 0).sum())}
    if L == 0:
        return X, info

    pn = np.array(plane_n)
    pd = np.array(plane_d)
    cv = F.ravel()
    cl = np.repeat(labels, 3)
    m = cl >= 0
    key = cv[m].astype(np.int64) * L + cl[m]
    ukey = np.unique(key)
    uv = ukey // L
    ul = (ukey % L).astype(np.int64)
    verts, inv = np.unique(uv, return_inverse=True)
    mV = len(verts)
    Nn, Dd = pn[ul], pd[ul]

    A = np.zeros((mV, 3, 3))
    b = np.zeros((mV, 3))
    for r in range(3):
        for c in range(3):
            A[:, r, c] = np.bincount(inv, weights=Nn[:, r] * Nn[:, c], minlength=mV)
        b[:, r] = np.bincount(inv, weights=Nn[:, r] * Dd, minlength=mV)
    eps = 2e-2  # تنظيم: يمنع القفز بعيدًا عند التقاء مستويات شبه متوازية (تقاطعها سيّئ الشرط)
    x0 = X[verts]
    A += eps * np.eye(3)
    sol = np.linalg.solve(A, (b + eps * x0)[:, :, None])[:, :, 0]

    touch_non = np.zeros(nV, dtype=bool)
    touch_non[cv[cl < 0]] = True
    w = np.where(touch_non[verts], boundary_w, 1.0)
    delta = (sol - x0) * w[:, None]
    dn = np.linalg.norm(delta, axis=1)
    sc = np.minimum(1.0, (2.0 * plane_tol) / np.maximum(dn, 1e-30))
    X2 = X.copy()
    X2[verts] = x0 + delta * sc[:, None]
    info["n_vertices_regularized"] = int(mV)
    return X2, info


# =============================================================================
# خريطة الانحراف والألوان
# =============================================================================

def jet(t):
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=1)


def nearest_neighbor_indices(query_points, reference_points):
    query_points = np.asarray(query_points)
    reference_points = np.asarray(reference_points)
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(reference_points).query(query_points, k=1, workers=-1)
        return idx
    except ImportError:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(reference_points)
        tree = o3d.geometry.KDTreeFlann(pc)
        idx = np.empty(len(query_points), dtype=np.int64)
        for i, p in enumerate(query_points):
            _f, nn, _d = tree.search_knn_vector_3d(p, 1)
            idx[i] = nn[0]
        return idx


# =============================================================================
# المستويات الجاهزة (قابلة للتجاوز من سطر الأوامر)
# =============================================================================

LEVELS = {
    "light":  dict(outer=1, n_iters=5,  v_iters=5,  feature_angle=25.0, disp_factor=0.5, fill_iters=6),
    "medium": dict(outer=2, n_iters=10, v_iters=10, feature_angle=30.0, disp_factor=1.0, fill_iters=12),
    "strong": dict(outer=3, n_iters=15, v_iters=15, feature_angle=35.0, disp_factor=2.0, fill_iters=20),
}


def refine_geometry(V0, F, topo, edges, deg, p, fill_soft, log_fn=log):
    X = V0.copy()
    nV = len(V0)
    maxd = np.full(nV, p["max_disp"], dtype=np.float64)
    if fill_soft is not None:
        maxd = maxd * (1.0 + (p["fill_boost"] - 1.0) * fill_soft)
    sigma_r = max(1.0 - math.cos(math.radians(p["feature_angle"])), 1e-6)

    for k in range(p["outer"]):
        t0 = time.time()
        Nf, Af, _Cf = face_geometry(X, F)
        Nfilt = filter_normals(Nf, Af, topo["Pi"], topo["Pj"], sigma_r, p["n_iters"])
        X = update_vertices(X, F, Nfilt, p["v_iters"], p["step"])
        X = clamp_displacement(X, V0, maxd)
        log_fn(f"    · دورة تنعيم {k + 1}/{p['outer']} ({human_time(time.time() - t0)})")

    if fill_soft is not None and p["fill_iters"] > 0:
        t0 = time.time()
        X = taubin_masked(X, edges, deg, fill_soft, p["fill_iters"])
        X = clamp_displacement(X, V0, maxd)
        log_fn(f"    · تنعيم إضافي لمناطق الإكمال ({human_time(time.time() - t0)})")
    return X


def parse_levels(s):
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x not in LEVELS:
            raise argparse.ArgumentTypeError(f"مستوى غير معروف: {x} (المتاح: {', '.join(LEVELS)})")
        if x not in out:
            out.append(x)
    if not out:
        raise argparse.ArgumentTypeError("لم يُحدَّد أي مستوى")
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="تحسين سطح ناتج عن بواسون: تنعيم يحافظ على الحواف + تنظيم المستويات اختياريًا + خريطة انحراف.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, help="ملف السطح المدخل (ply/obj/stl ...)، يُفضَّل ناتج سكربت بواسون")
    p.add_argument("-o", "--output", default=None,
                   help="مسار/اسم الملف الناتج (.ply). افتراضيًا: <المدخل>_refined.ply. "
                        "عند عدة مستويات يُضاف اسم المستوى قبل الامتداد")
    p.add_argument("--level", type=parse_levels, default=["medium"],
                   help="قوة التنعيم: light, medium, strong أو عدة مستويات مفصولة بفواصل لإنتاج ملف لكل مستوى")
    p.add_argument("--feature-angle", type=float, default=None,
                   help="زاوية الميزة بالدرجات: فروقات اتجاه أكبر منها تُعتبر حافة حقيقية ولا يُخلَط عبرها. "
                        "أصغر = حفاظ أدق على التفاصيل (وضجيج أكثر يبقى). افتراضيًا بحسب المستوى")
    p.add_argument("--max-disp", type=float, default=None,
                   help="أقصى إزاحة مسموحة لأي رأس عن موضعه الأصلي (بوحدة الملف، مثلًا متر). "
                        "افتراضيًا: عامل المستوى × وسيط طول الضلع")
    p.add_argument("--step", type=float, default=0.5, help="عامل خطوة تحريك الرؤوس (0-1)؛ أصغر = أثبت")
    p.add_argument("--fill-color", nargs=3, type=int, default=[230, 30, 30], metavar=("R", "G", "B"),
                   help="لون تمييز مناطق الإكمال الذي استخدمه سكربت بواسون (0-255)")
    p.add_argument("--fill-boost", type=float, default=3.0,
                   help="مضاعف سقف الإزاحة في مناطق الإكمال (تُنعَّم بقوة أكبر لأنها بلا بيانات قياس)")
    p.add_argument("--no-fill-smoothing", action="store_true", help="تعطيل المعاملة الخاصة لمناطق الإكمال")
    p.add_argument("--planes", action="store_true", help="تفعيل كشف المناطق المستوية وتنظيمها (جدران، فتحات)")
    p.add_argument("--plane-angle", type=float, default=3.0, help="أقصى انحراف اتجاه (درجات) عن المستوى ليُعتبر المثلث منه")
    p.add_argument("--plane-min-width", type=float, default=None,
                   help="أقل عرض (بوحدة الملف) لمنطقة كي تُعتبر مستوية؛ يرفض شرائح الأعمدة والقباب الرفيعة. "
                        "افتراضيًا: 8 × وسيط طول الضلع")
    p.add_argument("--plane-tol", type=float, default=None,
                   help="تسامح المستوى بوحدة الملف (أقصى بعد عن المستوى ليُعتبر الوجه منه). "
                        "افتراضيًا: 1 × وسيط طول الضلع — يُطبع بالسجل لتعايره")
    p.add_argument("--plane-min-faces", type=int, default=None,
                   help="أقل عدد مثلثات لمنطقة كي تُعتبر مستوية (افتراضيًا: 0.1%% من المثلثات، وحد أدنى 200)")
    p.add_argument("--plane-boundary-weight", type=float, default=0.5,
                   help="وزن سحب الرؤوس الحدودية بين منطقة مستوية وأخرى غير مستوية (0-1)")
    p.add_argument("--no-deviation-map", action="store_true", help="عدم كتابة ملف خريطة الانحراف")
    p.add_argument("--dev-max", type=float, default=None,
                   help="الانحراف الذي يقابل اللون الأحمر في الخريطة (افتراضيًا: المئين 95)")
    p.add_argument("--simplify-to", type=int, default=None, metavar="N",
                   help="كتابة نسخة إضافية مبسّطة إلى N مثلث (بعد التنعيم)، بملف منفصل *_simplified.ply")
    p.add_argument("--visualize", choices=["none", "result", "deviation"], default="none",
                   help="عرض ثلاثي الأبعاد لآخر مستوى منتَج")
    p.add_argument("--max-ram-percent", type=float, default=80.0,
                   help="إن قدّر السكربت أن الذاكرة المطلوبة تتجاوز هذه النسبة من الرام المتاح يتوقف (إلا مع --force)")
    p.add_argument("--force", action="store_true", help="تجاوز فحص الذاكرة التقديري")
    return p


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main():
    args = build_arg_parser().parse_args()
    if not os.path.isfile(args.input):
        log(f"خطأ: الملف غير موجود: {args.input}")
        sys.exit(1)

    log(f"➤ قراءة السطح: {args.input}")
    mesh = o3d.io.read_triangle_mesh(args.input)
    V0 = np.asarray(mesh.vertices, dtype=np.float64).copy()
    F = np.asarray(mesh.triangles, dtype=np.int64).copy()
    if len(V0) == 0 or len(F) == 0:
        log("خطأ: السطح فارغ أو لا يحتوي مثلثات.")
        sys.exit(1)
    colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
    nV, nF = len(V0), len(F)
    log(f"— رؤوس: {nV:,} | مثلثات: {nF:,} | ألوان رؤوس: {'نعم' if colors is not None else 'لا'}")

    # فحص تقديري للذاكرة (تقريب: ~ 1.2 كيلوبايت لكل مثلث أثناء التنعيم)
    est_gb = nF * 1200 / 1024.0 ** 3
    mem = get_mem_info()
    if mem.get("MemAvailable"):
        limit_gb = mem["MemAvailable"] * args.max_ram_percent / 100.0
        log(f"— ذاكرة تقديرية مطلوبة ≈ {est_gb:.2f} GB | المتاح الآن {mem['MemAvailable']:.1f} GB")
        if est_gb > limit_gb and not args.force:
            log("✗ الذاكرة التقديرية تتجاوز الحد الآمن. بسّط السطح أولًا (أو استخدم --force إن كنت متأكدًا).")
            sys.exit(2)

    t0 = time.time()
    topo = build_topology(F, nV)
    edges = topo["edges"]
    deg = (np.bincount(edges[:, 0], minlength=nV) + np.bincount(edges[:, 1], minlength=nV)).astype(np.float64)
    med_edge = median_edge_length(V0, edges)
    log(f"— وسيط طول الضلع: {med_edge:.6g} (بوحدة الملف) | بناء الطوبولوجيا: {human_time(time.time() - t0)}")

    # مناطق الإكمال من لون التمييز
    fill_soft = None
    if colors is not None and not args.no_fill_smoothing:
        fc = np.array(args.fill_color, dtype=np.float64) / 255.0
        fill = np.all(np.abs(colors - fc) < (1.5 / 255.0), axis=1)
        if fill.any():
            m = fill.astype(np.float64)
            for _ in range(4):  # تلاشي ناعم عند الحدود بدل قفزة حادة
                m = np.maximum(fill.astype(np.float64), 0.5 * m + 0.5 * vertex_neighbor_mean(m, edges, np.maximum(deg, 1)))
            fill_soft = np.clip(m, 0.0, 1.0)
            log(f"— مناطق الإكمال المكتشفة (لون التمييز): {int(fill.sum()):,} رأس ({100.0 * fill.mean():.1f}%)")
        else:
            log("— لم يُعثَر على لون التمييز في الملف؛ سيُنعَّم السطح بلا معاملة خاصة لمناطق الإكمال.")

    base_out = args.output
    if base_out is None:
        b, _ = os.path.splitext(os.path.abspath(args.input))
        base_out = b + "_refined.ply"
    base_out = os.path.abspath(base_out)
    os.makedirs(os.path.dirname(base_out) or ".", exist_ok=True)
    stem, ext = os.path.splitext(base_out)
    if ext.lower() != ".ply":
        ext = ".ply"
        base_out = stem + ext

    results = []
    for level in args.level:
        p = dict(LEVELS[level])
        if args.feature_angle is not None:
            p["feature_angle"] = args.feature_angle
        p["max_disp"] = args.max_disp if args.max_disp is not None else p["disp_factor"] * med_edge
        p["step"] = args.step
        p["fill_boost"] = args.fill_boost

        log("")
        log(f"====== مستوى: {level} (زاوية الميزة {p['feature_angle']}°، سقف الإزاحة {p['max_disp']:.4g}) ======")
        t1 = time.time()
        X = refine_geometry(V0, F, topo, edges, deg, p, fill_soft)

        plane_info = None
        if args.planes:
            plane_tol = args.plane_tol if args.plane_tol is not None else 1.0 * med_edge
            min_faces = args.plane_min_faces if args.plane_min_faces is not None else max(200, int(0.001 * nF))
            log(f"    · كشف المستويات (تسامح {plane_tol:.4g}، أقل عدد مثلثات {min_faces:,}، أقل عرض {(args.plane_min_width if args.plane_min_width is not None else 8.0 * med_edge):.4g})...")
            t2 = time.time()
            min_width = args.plane_min_width if args.plane_min_width is not None else 8.0 * med_edge
            X, plane_info = regularize_planes(X, F, topo, args.plane_angle, plane_tol, min_faces,
                                              args.plane_boundary_weight, min_width=min_width)
            log(f"    · عدد المستويات المكتشفة: {plane_info['n_planes']} "
                f"({plane_info['n_planar_faces']:,} مثلث مستوٍ من {nF:,}) ({human_time(time.time() - t2)})")

        dev = np.linalg.norm(X - V0, axis=1)
        out_path = base_out if len(args.level) == 1 else f"{stem}_{level}{ext}"

        out_mesh = o3d.geometry.TriangleMesh()
        out_mesh.vertices = o3d.utility.Vector3dVector(X)
        out_mesh.triangles = o3d.utility.Vector3iVector(F)
        if colors is not None:
            out_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        out_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(out_path, out_mesh, write_vertex_colors=colors is not None)

        dev_path = None
        dev_mesh = None
        dmax = args.dev_max if args.dev_max is not None else float(np.percentile(dev, 95))
        if not args.no_deviation_map:
            dev_mesh = o3d.geometry.TriangleMesh(out_mesh)
            dev_mesh.vertex_colors = o3d.utility.Vector3dVector(jet(dev / max(dmax, 1e-30)))
            dev_path = os.path.splitext(out_path)[0] + "_deviation.ply"
            o3d.io.write_triangle_mesh(dev_path, dev_mesh, write_vertex_colors=True)

        simp_path = None
        if args.simplify_to:
            simp = out_mesh.simplify_quadric_decimation(target_number_of_triangles=int(args.simplify_to))
            if colors is not None and len(simp.vertices) > 0:
                idx = nearest_neighbor_indices(np.asarray(simp.vertices), X)
                simp.vertex_colors = o3d.utility.Vector3dVector(colors[idx])
            simp.compute_vertex_normals()
            simp_path = os.path.splitext(out_path)[0] + "_simplified.ply"
            o3d.io.write_triangle_mesh(simp_path, simp, write_vertex_colors=colors is not None)

        stats = {
            "level": level,
            "elapsed_sec": time.time() - t1,
            "params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in p.items()},
            "displacement": {
                "mean": float(dev.mean()), "p95": float(np.percentile(dev, 95)),
                "max": float(dev.max()), "dev_map_red_at": float(dmax),
            },
            "planes": plane_info,
            "files": {"mesh": out_path, "deviation_map": dev_path, "simplified": simp_path},
            "n_vertices": nV, "n_triangles": nF,
        }
        rep_path = os.path.splitext(out_path)[0] + "_refine_report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        log(f"  ✓ الزمن: {human_time(stats['elapsed_sec'])}")
        log(f"  ✓ انحراف الرؤوس عن الأصل — متوسط: {dev.mean():.4g} | المئين 95: {np.percentile(dev, 95):.4g} | أقصى: {dev.max():.4g}")
        log("    (إن كانت وحدة الملف متر فاضرب بـ1000 للحصول على الملم)")
        log(f"  ✓ الملف: {out_path}")
        if dev_path:
            log(f"  ✓ خريطة الانحراف (أحمر = {dmax:.4g} فأكثر): {dev_path}")
        if simp_path:
            log(f"  ✓ نسخة مبسّطة: {simp_path}")
        log(f"  ✓ التقرير: {rep_path}")
        results.append((out_mesh, dev_mesh))

    if args.visualize != "none" and results:
        if sys.platform.startswith("linux") and not _has_display():
            log("(تخطّي العرض: لا توجد بيئة رسومية DISPLAY)")
        else:
            m, d = results[-1]
            target = m if args.visualize == "result" else (d if d is not None else m)
            try:
                o3d.visualization.draw_geometries([target], window_name="refined mesh")
            except Exception as e:  # noqa: BLE001
                log(f"تعذّر فتح نافذة العرض: {e}")


if __name__ == "__main__":
    main()
