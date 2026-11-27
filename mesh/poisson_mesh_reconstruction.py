#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poisson_mesh_reconstruction.py
================================
تحويل سحابة نقطية (Point Cloud) إلى سطح (Mesh) باستخدام خوارزمية بواسون
(Poisson Surface Reconstruction) عبر مكتبة Open3D، مصمَّم للعمل بأمان وكفاءة
على أجهزة ذات موارد محدودة (مثال: رام 20GB + معالج Core i7 من الجيل الثامن).

الميزات الرئيسية
----------------
1) تحليل تلقائي للسحابة النقطية (عدد النقاط، توفر الألوان/الأنماط، متوسط
   التباعد بين النقاط، أبعاد صندوق الإحاطة) لاقتراح قيمة depth مناسبة.
2) تقدير زمني تقريبي *قبل* التنفيذ الفعلي عبر معايرة بعيّنتين مصغّرتين من
   نفس السحابة على نفس الجهاز (ملاءمة قانون قوة Time ≈ a·N^b)، مع إمكانية
   مقارنة عدة قيم depth دفعة واحدة (--depth 8,9,10).
3) تنفيذ إعادة البناء الفعلية داخل عملية فرعية منفصلة تتم مراقبة استهلاكها
   الفعلي للذاكرة لحظيًا، ويتم إيقافها قسريًا إن تجاوزت حدًا آمنًا من رام
   الجهاز — وهي خط الأمان الحقيقي (بصرف النظر عن أي تقدير مسبق).
4) ضبط تلقائي لعدد الخيوط (threads) بحسب عدد الأنوية المتاحة مع ترك هامش
   للنظام، دون الحاجة لأي مكتبات خارجية غير Open3D وNumPy (القراءة من
   /proc على لينكس مباشرة).
5) تمييز المناطق التي "أكملتها"/استنتجتها خوارزمية بواسون في أماكن نقص
   البيانات (أي الرؤوس ذات قيمة density المنخفضة) بلون مغاير قابل للتخصيص،
   مع محاولة نقل الألوان الأصلية من السحابة النقطية إلى بقية السطح.
6) خيارات كاملة لمسارات الإدخال/الإخراج، اسم الملف الناتج، وأوضاع العرض
   ثلاثي الأبعاد (أو حفظ لقطة شاشة عند العمل بلا بيئة رسومية عبر SSH).

المتطلبات
---------
    pip install --break-system-packages open3d numpy
    # اختياري وموصى به لتسريع نقل الألوان الأصلية إلى السطح الناتج:
    pip install --break-system-packages scipy

أمثلة الاستخدام
----------------
    # تشغيل تلقائي كامل بالقيم الافتراضية (depth مقترح تلقائيًا + تأكيد تفاعلي)
    python3 poisson_mesh_reconstruction.py -i scan.ply

    # تحديد مسار/اسم ملف الإخراج كما يريد المستخدم
    python3 poisson_mesh_reconstruction.py -i scan.ply -o /home/user/results/my_mesh.ply

    # مقارنة عدة قيم depth (زمن تقديري فقط) دون أي تنفيذ فعلي
    python3 poisson_mesh_reconstruction.py -i scan.ply --depth 8,9,10,11 --compare-only

    # تنفيذ مباشر بقيمة depth محددة دون تأكيد تفاعلي، مع سقف ذاكرة 70%
    python3 poisson_mesh_reconstruction.py -i scan.ply --depth 10 --yes --max-ram-percent 70

    # بيئة بلا شاشة (SSH بلا X11): تعطيل العرض التفاعلي وحفظ لقطة شاشة بدلاً منه
    python3 poisson_mesh_reconstruction.py -i scan.ply --visualize none --screenshot preview.png
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# ضبط عدد الخيوط (threads) للمكتبات العددية *قبل* استيراد numpy/open3d قدر
# الإمكان، لأن أغلب تطبيقات OpenMP/BLAS تقرأ هذه المتغيرات عند أول استخدام
# فعلي للمعالجة المتوازية. هذا "تحسين قدر الإمكان" وليس ضمانًا كاملًا —
# إن كانت مكتبة BLAS المُحمَّلة تهيّئ خيوطها عند التحميل لا عند أول استخدام،
# فقد لا يكون للأمر أثر كامل؛ خط الأمان الحقيقي هو مراقبة الذاكرة أدناه.
# ---------------------------------------------------------------------------
def _preparse_threads():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--threads", type=int, default=None)
    ns, _ = pre.parse_known_args()
    return ns.threads


_CPU_TOTAL = os.cpu_count() or 4
_THREADS_DEFAULT = max(1, _CPU_TOTAL - 2) if _CPU_TOTAL > 2 else max(1, _CPU_TOTAL - 1)
_THREADS_EARLY = _preparse_threads() or _THREADS_DEFAULT
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(_THREADS_EARLY))

import json
import math
import time
import traceback
import multiprocessing
import numpy as np
import open3d as o3d


# =============================================================================
# أدوات قراءة معلومات النظام (ذاكرة/عمليات) مباشرة من /proc — خاص بلينكس،
# يناسب تمامًا بيئة Linux Mint المستهدفة ودون الحاجة لمكتبة psutil.
# =============================================================================

def get_mem_info():
    """يعيد {'MemTotal': GB, 'MemAvailable': GB} بقراءة /proc/meminfo، أو {} إن تعذّر ذلك."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    kb = int(rest.strip().split()[0])
                    info[key] = kb / (1024.0 ** 2)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return info


def get_rss_mb(pid):
    """يعيد استهلاك الذاكرة الفعلي (RSS) بالميغابايت لعملية معيّنة، أو None إن تعذّر ذلك."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None
    return None


def human_time(seconds):
    if seconds is None:
        return "غير معروف"
    if seconds < 60:
        return f"{seconds:.1f} ثانية"
    if seconds < 3600:
        return f"{seconds / 60:.1f} دقيقة"
    return f"{seconds / 3600:.2f} ساعة"


def log(msg=""):
    print(msg, flush=True)


# =============================================================================
# تحليل السحابة النقطية وتقدير depth المناسب
# =============================================================================

def compute_avg_spacing(pcd, sample_k=2000, seed=0):
    """متوسط تقريبي للمسافة بين كل نقطة وأقرب جارة لها، بعيّنة عشوائية (وليس كل النقاط)."""
    n = len(pcd.points)
    if n < 2:
        return 0.0
    k = min(sample_k, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in idx:
        found, _nn_idx, nn_dist2 = tree.search_knn_vector_3d(pts[i], 2)
        if found >= 2:
            dists.append(math.sqrt(max(nn_dist2[1], 0.0)))
    return float(np.mean(dists)) if dists else 0.0


def recommend_depth(n_points, available_ram_gb):
    """اقتراح أولي لقيمة depth بحسب عدد النقاط، يُخفَّض احترازيًا إذا كانت الذاكرة المتاحة محدودة.
    هذا مجرد نقطة بداية تُتحقَّق لاحقًا عبر المعايرة الزمنية، وليس قيمة نهائية مضمونة."""
    if n_points < 50_000:
        base = 7
    elif n_points < 300_000:
        base = 8
    elif n_points < 1_500_000:
        base = 9
    elif n_points < 6_000_000:
        base = 10
    elif n_points < 15_000_000:
        base = 11
    else:
        base = 12

    if available_ram_gb and available_ram_gb < 8 and base > 8:
        base -= 1
    if available_ram_gb and available_ram_gb < 4 and base > 7:
        base -= 1
    return max(6, min(base, 12))


def calibrate_depth_timing(pcd_full, depth, cfg, logger=log):
    """معايرة زمنية تقريبية: تشغيل بواسون فعليًا على عيّنتين صغيرتين من نفس السحابة
    بنفس قيمة depth، ثم ملاءمة قانون قوة Time ≈ a * N^b لتوقع زمن السحابة كاملة.
    هذا تقدير تقريبي من هذا الجهاز بالذات على جزء صغير من البيانات، وليس رقمًا
    مضمونًا — خصوصًا أن تكلفة الحل عند depth ثابت لا تتناسب خطيًا بالضرورة مع N."""
    n_total = len(pcd_full.points)
    if n_total < 500:
        return None

    avg_sp = cfg["avg_spacing"] if cfg["avg_spacing"] > 0 else 1e-3
    radius = avg_sp * 3.0

    sizes = []
    for frac in (cfg["calib_frac_1"], cfg["calib_frac_2"]):
        s = int(min(max(n_total * frac, cfg["calib_min"]), cfg["calib_max"], n_total))
        sizes.append(max(s, 1))
    sizes = sorted(set(sizes))
    if len(sizes) < 2:
        extra = max(10, int(n_total * 0.5))
        sizes = sorted(set(sizes + [min(extra, n_total)]))
    if len(sizes) < 2:
        return None

    measured = []
    for s in sizes:
        ratio = s / n_total
        if ratio >= 0.999:
            sample = o3d.geometry.PointCloud(pcd_full)
        else:
            sample = pcd_full.random_down_sample(ratio)
        if len(sample.points) < 10:
            continue
        if not sample.has_normals():
            sample.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
        t0 = time.time()
        try:
            _mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sample, depth=depth)
        except Exception as e:  # noqa: BLE001
            logger(f"  (تعذّر اختبار المعايرة عند عيّنة بحجم {len(sample.points):,} لـ depth={depth}: {e})")
            continue
        measured.append((len(sample.points), time.time() - t0))

    if len(measured) < 2:
        return None

    (n1, t1), (n2, t2) = measured[0], measured[-1]
    t1 = max(t1, 1e-3)
    t2 = max(t2, 1e-3)
    if n1 == n2:
        b = 1.0
    else:
        b = math.log(t2 / t1) / math.log(n2 / n1)
    a = t1 / (n1 ** b)
    predicted = a * (n_total ** b)
    return {
        "measured": measured,
        "exponent": b,
        "coefficient": a,
        "predicted_seconds": predicted,
    }


# =============================================================================
# منطق التنفيذ الفعلي (يعمل داخل عملية فرعية منفصلة قابلة للمراقبة والإيقاف)
# =============================================================================

def _colorize_mesh(mesh, densities, source_pcd, cfg):
    """تلوين السطح الناتج: نقل الألوان الأصلية من السحابة النقطية للمناطق المدعومة
    جيدًا بالبيانات، وتمييز المناطق ذات الكثافة المنخفضة (التي استنتجتها/أكملتها
    خوارزمية بواسون في أماكن نقص البيانات) بلون مغاير محدَّد من المستخدم."""
    n = len(densities)
    threshold = float(np.percentile(densities, cfg["fill_percentile"])) if n else 0.0
    is_filled = densities < threshold

    colors = np.tile(np.array(cfg["base_color"], dtype=np.float64), (max(n, 1), 1))
    transferred = False

    if n and source_pcd.has_colors() and cfg["transfer_colors"]:
        try:
            from scipy.spatial import cKDTree  # noqa: WPS433
            src_points = np.asarray(source_pcd.points)
            src_colors = np.asarray(source_pcd.colors)
            verts = np.asarray(mesh.vertices)
            tree = cKDTree(src_points)
            _, idx = tree.query(verts, k=1, workers=-1)
            good = ~is_filled
            colors[good] = src_colors[idx[good]]
            transferred = True
        except ImportError:
            pass  # سيُستخدم اللون الأساسي بدلاً من نقل الألوان الحقيقية (انظر الرسالة في main)

    if n:
        colors[is_filled] = np.array(cfg["fill_color"], dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh, {
        "threshold": threshold,
        "n_filled": int(is_filled.sum()),
        "colors_transferred": transferred,
    }


def _run_reconstruction_worker(cfg):
    """تُنفَّذ هذه الدالة داخل عملية فرعية (multiprocessing, start method = spawn) كي
    يمكن مراقبة استهلاكها الفعلي للذاكرة من العملية الأم وإيقافها قسرًا عند الحاجة
    دون التأثير على العملية الرئيسية. تقرأ السحابة من القرص مجددًا لتفادي أي تعقيد
    متعلق بنقل كائنات Open3D بين العمليات."""
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = str(cfg["threads"])

    report = {"status": "running"}
    t0 = time.time()
    try:
        pcd = o3d.io.read_point_cloud(cfg["input_path"])
        if len(pcd.points) == 0:
            raise RuntimeError("تعذّرت قراءة السحابة النقطية داخل العملية الفرعية (ناتج فارغ).")

        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=cfg["normal_radius"], max_nn=30)
            )
            if cfg["orient_normals"]:
                pcd.orient_normals_consistent_tangent_plane(k=15)
        t_normals = time.time() - t0

        t1 = time.time()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=cfg["depth"])
        t_poisson = time.time() - t1
        densities = np.asarray(densities)

        mesh, color_stats = _colorize_mesh(mesh, densities, pcd, cfg)
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(cfg["output_path"], mesh, write_vertex_colors=True)

        report.update({
            "status": "done",
            "elapsed_normals_sec": t_normals,
            "elapsed_poisson_sec": t_poisson,
            "elapsed_total_sec": time.time() - t0,
            "n_vertices": len(mesh.vertices),
            "n_triangles": len(mesh.triangles),
            "density_min": float(densities.min()) if len(densities) else 0.0,
            "density_max": float(densities.max()) if len(densities) else 0.0,
            "density_threshold": color_stats["threshold"],
            "n_filled_vertices": color_stats["n_filled"],
            "colors_transferred": color_stats["colors_transferred"],
            "output_path": cfg["output_path"],
        })
    except Exception:  # noqa: BLE001
        report.update({
            "status": "error",
            "message": str(sys.exc_info()[1]),
            "traceback": traceback.format_exc(),
        })

    with open(cfg["report_path"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report["status"] == "error":
        sys.exit(1)


# =============================================================================
# العرض ثلاثي الأبعاد / لقطات الشاشة
# =============================================================================

def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _safe_visualize(geoms, title):
    if sys.platform.startswith("linux") and not _has_display():
        log(f"(تخطّي العرض التفاعلي لـ «{title}»: لا توجد بيئة عرض رسومي DISPLAY. "
            f"استخدم --screenshot عند العمل عبر SSH بلا واجهة رسومية.)")
        return
    try:
        o3d.visualization.draw_geometries(list(geoms), window_name=title)
    except Exception as e:  # noqa: BLE001
        log(f"تعذّر فتح نافذة العرض: {e}")


def _save_screenshot(mesh, path):
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path, do_render=True)
        vis.destroy_window()
        log(f"— تم حفظ لقطة شاشة للسطح الناتج في: {path}")
    except Exception as e:  # noqa: BLE001
        log(f"تعذّر حفظ لقطة الشاشة: {e}")


# =============================================================================
# سطر الأوامر
# =============================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="تحويل سحابة نقطية إلى سطح (Mesh) عبر خوارزمية بواسون باستخدام Open3D، "
                     "مع إدارة ديناميكية للموارد وتمييز المناطق المُكمَّلة بلون مغاير.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True,
                   help="مسار ملف السحابة النقطية المدخلة (ply, pcd, xyz, xyzrgb, pts ...)")
    p.add_argument("-o", "--output", default=None,
                   help="مسار/اسم ملف السطح الناتج (.ply يُفضَّل لدعمه ألوان الرؤوس). "
                        "افتراضيًا: بجانب ملف الإدخال باسم <اسم الإدخال>_poisson_mesh.ply")
    p.add_argument("--depth", default="auto",
                   help="عمق بواسون: رقم واحد، أو 'auto' للاقتراح التلقائي، "
                        "أو أرقام مفصولة بفواصل للمقارنة بينها مثل 8,9,10")
    p.add_argument("--compare-only", action="store_true",
                   help="عرض جدول مقارنة الأزمنة التقديرية لقيم depth المطروحة فقط دون تنفيذ فعلي")
    p.add_argument("-y", "--yes", action="store_true",
                   help="المتابعة دون طلب تأكيد تفاعلي (مناسب للتشغيل غير التفاعلي)")
    p.add_argument("--threads", type=int, default=None,
                   help="عدد الخيوط المستخدمة (افتراضيًا: عدد الأنوية المنطقية ناقص 2، بحد أدنى 1)")
    p.add_argument("--max-ram-percent", type=float, default=80.0,
                   help="أقصى نسبة من إجمالي رام الجهاز يُسمح للعملية الفعلية باستهلاكها قبل إيقافها قسرًا")
    p.add_argument("--fill-percentile", type=float, default=10.0,
                   help="النسبة المئوية الدنيا لقيم density التي تُعتبر دونها المنطقة 'مُكمَّلة/مُستنتجة'")
    p.add_argument("--fill-color", nargs=3, type=int, default=[230, 30, 30], metavar=("R", "G", "B"),
                   help="لون تمييز المناطق المُكمَّلة (0-255)")
    p.add_argument("--base-color", nargs=3, type=int, default=[200, 200, 210], metavar=("R", "G", "B"),
                   help="اللون الأساسي للمناطق المدعومة جيدًا إن لم تتوفر ألوان أصلية أو تعذّر نقلها")
    p.add_argument("--no-color-transfer", action="store_true",
                   help="تعطيل محاولة نقل الألوان الأصلية من السحابة النقطية إلى السطح الناتج")
    p.add_argument("--no-orient-normals", action="store_true",
                   help="تخطي توحيد اتجاه الأنماط الطبيعية (أسرع، مناسب فقط إن كانت موجَّهة مسبقًا باتساق)")
    p.add_argument("--visualize", choices=["none", "input", "result", "both"], default="result",
                   help="فتح نافذة عرض ثلاثي الأبعاد بعد الانتهاء")
    p.add_argument("--screenshot", default=None,
                   help="مسار لحفظ لقطة شاشة للسطح الناتج (مفيد عند العمل بلا بيئة رسومية)")
    p.add_argument("--calib-sample-fractions", nargs=2, type=float, default=[0.02, 0.08],
                   metavar=("F1", "F2"), help="نسبتا حجمي العيّنتين المستخدمتين في المعايرة الزمنية")
    p.add_argument("--calib-min-points", type=int, default=15000,
                   help="الحد الأدنى لعدد نقاط عيّنة المعايرة")
    p.add_argument("--calib-max-points", type=int, default=250000,
                   help="الحد الأقصى لعدد نقاط عيّنة المعايرة")
    p.add_argument("--poll-interval", type=float, default=2.0,
                   help="فترة فحص استهلاك الذاكرة أثناء التنفيذ الفعلي (بالثواني)")
    return p


# =============================================================================
# البرنامج الرئيسي
# =============================================================================

def main():
    args = build_arg_parser().parse_args()

    if not os.path.isfile(args.input):
        log(f"خطأ: ملف الإدخال غير موجود: {args.input}")
        sys.exit(1)

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(os.path.abspath(args.input))
        out_path = base + "_poisson_mesh.ply"
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    report_base, _ = os.path.splitext(out_path)
    report_path = report_base + "_report.json"

    log(f"➤ قراءة السحابة النقطية من: {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    n_points = len(pcd.points)
    if n_points == 0:
        log("خطأ: السحابة النقطية فارغة أو تعذّرت قراءتها (تحقّق من الامتداد/الصيغة).")
        sys.exit(1)

    mem = get_mem_info()
    total_ram = mem.get("MemTotal")
    avail_ram = mem.get("MemAvailable")
    threads = args.threads or _THREADS_DEFAULT

    log(f"— عدد النقاط: {n_points:,}")
    log(f"— ألوان متوفرة: {'نعم' if pcd.has_colors() else 'لا'} | أنماط طبيعية متوفرة: {'نعم' if pcd.has_normals() else 'لا'}")
    if total_ram:
        log(f"— ذاكرة النظام: {total_ram:.1f} GB إجمالي، {avail_ram:.1f} GB متاحة الآن")
    else:
        log("— تعذّرت قراءة معلومات الذاكرة من /proc (غير متاح على منصّات غير لينكس).")
    log(f"— عدد الأنوية المنطقية: {_CPU_TOTAL} (سيُستخدم منها {threads} خيط/خيوط)")

    log("➤ تقدير متوسط التباعد بين النقاط (عيّنة عشوائية من 2000 نقطة)...")
    avg_spacing = compute_avg_spacing(pcd)
    bbox_diag = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    log(f"— متوسط التباعد التقريبي: {avg_spacing:.6g} | قطر صندوق الإحاطة: {bbox_diag:.4g}")

    if str(args.depth).strip().lower() == "auto":
        rec = recommend_depth(n_points, avail_ram or 16.0)
        depth_candidates = [rec]
        log(f"➤ القيمة المقترحة تلقائيًا لـ depth: {rec} (يمكن تجاوزها عبر --depth)")
    else:
        try:
            depth_candidates = sorted({int(x.strip()) for x in str(args.depth).split(",") if x.strip()})
            if not depth_candidates:
                raise ValueError
        except ValueError:
            log("خطأ: صيغة --depth غير صحيحة. استخدم رقمًا، أو 'auto'، أو أرقامًا مفصولة بفواصل مثل 8,9,10")
            sys.exit(1)

    calib_cfg = {
        "avg_spacing": avg_spacing,
        "calib_frac_1": args.calib_sample_fractions[0],
        "calib_frac_2": args.calib_sample_fractions[1],
        "calib_min": args.calib_min_points,
        "calib_max": args.calib_max_points,
    }

    log("➤ تشغيل معايرة زمنية سريعة على عيّنات مصغّرة من السحابة...")
    results_table = [(d, calibrate_depth_timing(pcd, d, calib_cfg)) for d in depth_candidates]

    log("")
    log("====== جدول تقدير الزمن (تقريبي، مبني على معايرة بعيّنات صغيرة على هذا الجهاز) ======")
    for d, calib in results_table:
        if calib is None:
            log(f"  depth={d:<3}  تعذّر تقدير الزمن (سحابة صغيرة جدًا أو خطأ أثناء المعايرة)")
        else:
            log(f"  depth={d:<3}  الزمن المتوقع لكامل السحابة: ~{human_time(calib['predicted_seconds'])}"
                f"   (قياس من عيّنات بحجم {[m[0] for m in calib['measured']]})")
    log("ملاحظة: هذا تقدير تقريبي فقط من عيّنات صغيرة؛ الزمن الفعلي قد يختلف خصوصًا عند قيم depth")
    log("مرتفعة أو توزيع كثافة غير منتظم. المراقبة الحيّة لاستهلاك الذاكرة أدناه هي خط الأمان الفعلي.")
    log("====================================================================================")

    if args.compare_only:
        log("تم العرض بوضع المقارنة فقط (--compare-only)، لم يُنفَّذ أي إعادة بناء فعلية.")
        return

    if len(depth_candidates) > 1:
        if args.yes:
            chosen_depth = min(depth_candidates)
            log(f"➤ تم اختيار أصغر قيمة depth تلقائيًا (--yes): {chosen_depth}")
        else:
            chosen_depth = None
            while chosen_depth is None:
                sel = input(f"اختر قيمة depth للتنفيذ الفعلي من {depth_candidates}: ").strip()
                if sel.isdigit() and int(sel) in depth_candidates:
                    chosen_depth = int(sel)
                else:
                    log("قيمة غير صحيحة، أعد المحاولة.")
    else:
        chosen_depth = depth_candidates[0]
        if not args.yes:
            resp = input(f"المتابعة بتنفيذ إعادة البناء الفعلية عند depth={chosen_depth}? [Y/n]: ").strip().lower()
            if resp not in ("", "y", "yes", "نعم", "ن"):
                log("تم الإلغاء من قبل المستخدم.")
                return

    max_rss_mb = (total_ram or 16.0) * 1024.0 * (args.max_ram_percent / 100.0)
    normal_radius = avg_spacing * 3.0 if avg_spacing > 0 else 0.05

    worker_cfg = {
        "input_path": os.path.abspath(args.input),
        "output_path": out_path,
        "report_path": report_path,
        "depth": chosen_depth,
        "threads": threads,
        "normal_radius": normal_radius,
        "orient_normals": not args.no_orient_normals,
        "fill_percentile": args.fill_percentile,
        "fill_color": [c / 255.0 for c in args.fill_color],
        "base_color": [c / 255.0 for c in args.base_color],
        "transfer_colors": not args.no_color_transfer,
    }

    if worker_cfg["transfer_colors"] and pcd.has_colors():
        try:
            import scipy  # noqa: F401
        except ImportError:
            log("تنبيه: مكتبة scipy غير مثبّتة — سيُستخدم لون أساسي بدل نقل الألوان الأصلية "
                "إلى السطح. للتفعيل: pip install --break-system-packages scipy")

    log(f"➤ بدء التنفيذ الفعلي عند depth={chosen_depth} ضمن عملية فرعية مراقَبة "
        f"(حد الذاكرة الآمن: {max_rss_mb / 1024:.1f} GB)...")

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_run_reconstruction_worker, args=(worker_cfg,))
    t_start = time.time()
    proc.start()
    peak_rss = 0.0
    killed = False
    try:
        while proc.is_alive():
            time.sleep(args.poll_interval)
            rss = get_rss_mb(proc.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
                if rss > max_rss_mb:
                    log(f"⚠ تجاوز استهلاك الذاكرة الحد الآمن ({rss / 1024:.2f} GB > "
                        f"{max_rss_mb / 1024:.2f} GB) — سيتم إيقاف العملية الآن.")
                    proc.terminate()
                    time.sleep(1.5)
                    if proc.is_alive():
                        proc.kill()
                    killed = True
                    break
        proc.join()
    except KeyboardInterrupt:
        log("➤ تم طلب الإيقاف من المستخدم (Ctrl+C)، جارٍ إنهاء العملية الفرعية...")
        proc.terminate()
        proc.join()
        sys.exit(130)

    elapsed = time.time() - t_start

    if killed:
        log("✗ تم إيقاف العملية قسرًا لتجاوزها حد الذاكرة الآمن. جرّب قيمة depth أقل، أو خفّض كثافة")
        log("  السحابة مسبقًا (voxel downsampling)، أو ارفع --max-ram-percent إن كان الجهاز يحتمل ذلك.")
        sys.exit(2)

    if not os.path.isfile(report_path):
        log("✗ لم يُنتَج أي تقرير من العملية الفرعية؛ يُحتمل أنها توقفت بشكل غير متوقع.")
        sys.exit(2)

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    if report.get("status") != "done":
        log(f"✗ فشلت عملية إعادة البناء: {report.get('message')}")
        if report.get("traceback"):
            log(report["traceback"])
        sys.exit(2)

    log("")
    log("====== اكتملت إعادة البناء بنجاح ======")
    log(f"— الزمن الفعلي الكلي: {human_time(report['elapsed_total_sec'])}  "
        f"(أنماط: {human_time(report['elapsed_normals_sec'])}، بواسون: {human_time(report['elapsed_poisson_sec'])})")
    log(f"— الزمن الكلي المقاس من العملية الأم (شامل الإقلاع): {human_time(elapsed)}")
    log(f"— أقصى استهلاك ذاكرة فعلي لوحظ: {peak_rss / 1024:.2f} GB")
    log(f"— عدد رؤوس السطح: {report['n_vertices']:,} | عدد المثلثات: {report['n_triangles']:,}")
    log(f"— الرؤوس المُميَّزة كمناطق 'مُكمَّلة' (كثافة أقل من عتبة المئين {args.fill_percentile}٪): "
        f"{report['n_filled_vertices']:,}")
    log(f"— عتبة الكثافة المستخدمة: {report['density_threshold']:.4g} "
        f"(المدى الكلي: {report['density_min']:.4g} .. {report['density_max']:.4g})")
    log(f"— نقل الألوان الأصلية: {'تم' if report['colors_transferred'] else 'لم يتم (استُخدم لون أساسي)'}")
    log(f"— الملف الناتج: {out_path}")
    log(f"— تقرير تفصيلي (JSON): {report_path}")
    log("========================================")

    if args.visualize in ("input", "both"):
        _safe_visualize([pcd], "السحابة النقطية المدخلة")

    if args.visualize in ("result", "both") or args.screenshot:
        mesh_out = o3d.io.read_triangle_mesh(out_path)
        mesh_out.compute_vertex_normals()
        if args.screenshot:
            _save_screenshot(mesh_out, args.screenshot)
        if args.visualize in ("result", "both"):
            _safe_visualize([mesh_out], "السطح الناتج (اللون المميَّز = مناطق أكملتها الخوارزمية)")


if __name__ == "__main__":
    main()
