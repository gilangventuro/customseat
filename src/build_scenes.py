"""Generate per-element color masks for multiple chair photos (dummy multi-angle demo).

Pipeline per photo:
1. cv2.grabCut -> foreground alpha (separates chair from wall/floor).
2. Within foreground, classify each pixel to nearest reference color (Body / Panel / Accent).
3. Armrest = pixels inside a manually-specified box AND classified as Body (armrest shares
   the body's black color, so it can't be separated by color alone -> region box instead).
   Scenes without a usable armrest view simply omit the box and get no armrest zone.
4. Each zone mask is saved as one channel of an RGBA PNG (R=body, G=panel, B=accent, A=armrest),
   plus a grayscale luminance JPG of the original photo (for shading-preserving recolor).
"""
import json, pathlib
import numpy as np
import cv2
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "src/scenes/raw"
OUT = ROOT / "src/scenes/out"
OUT.mkdir(parents=True, exist_ok=True)

ZONE_ORDER = ["body", "panel", "accent"]  # -> R,G,B channels; armrest -> A channel

# Classifying zones by nearest-RGB-reference-color turned out fragile: across the 4 photos
# (different exposure/white-balance per shot), shadowed folds in the white mesh fabric sit
# numerically closer to the teal accent reference than to the actual panel/body references,
# so whole swaths of mesh got misclassified as "accent". Measured directly from the photos
# (see analysis), the true teal stripe has a very tight, scene-independent hue band (~80-98
# in OpenCV's 0-179 hue) with decent saturation, while shadow/highlight noise is desaturated
# (low S) regardless of its apparent hue. So: classify accent by hue+saturation first (that's
# the one zone with a real, distinct color), then split whatever's left into body/panel purely
# by brightness (V) -- the two are black vs. off-white, which is an extremely clean split.
ACCENT_HUE = (75, 105)
ACCENT_SAT_MIN = 35
ACCENT_VAL_MIN = 45
BODY_VAL_MAX = 95

SCENES = {
    "depan":    {"rect_pad": (0.04, 0.02, 0.93, 0.97), "armrest_boxes": [(90,660,320,840), (650,660,880,840)]},
    "samping":  {
        "rect_pad": (0.06, 0.20, 0.88, 0.94),
        "armrest_boxes": [(420,600,680,715)],
        # samping has wall/floor tonally close to the chair, and the chair sits on a
        # diagonal (leaning back-left to front-right), so a single bounding rect always
        # includes a lot of genuine wall/floor off to the chair's left. Instead give
        # grabCut several narrow probable-foreground bands that hug the actual silhouette
        # (picked by eye from a coordinate-grid overlay of the photo), plus a couple of
        # small definite-foreground/definite-background stamps to anchor the color model.
        "seeds": {
            "pr_fg": [
                (80,260,220,420),
                (90,420,330,600),
                (130,600,720,800),
                (220,800,880,1000),
                (220,1000,760,1220),
            ],
            "fg": [(120,330,210,480)],
            "bg": [(0,0,960,230), (760,0,960,900), (0,850,260,1280), (760,900,960,1280)],
        },
    },
    "belakang": {
        # cut the rect off above the wheelbase/floor line -- the star-shaped plastic base
        # has gaps that expose floor, which previously got swept into the foreground together
        # with the cushion above it.
        "rect_pad": (0.16, 0.23, 0.80, 0.76),
        "armrest_boxes": [],
    },
    "atas":     {
        "rect_pad": (0.10, 0.08, 0.80, 0.98),
        # only the right armrest is actually visible in this photo; the left "armrest"
        # box from the first pass was really just floor misread as chair.
        "armrest_boxes": [(680,780,860,1010)],
        # the stone step/ledge to the chair's right, and the open floor to its left, are
        # dark/toned enough to get misread as chair body -- mark both explicitly background.
        "seeds": {"bg": [(690,600,960,1120), (0,550,330,1280)]},
    },
}

def grabcut_fg(bgr, pad, seeds=None):
    h, w = bgr.shape[:2]
    # grabCut is noticeably more reliable at a smaller, standardized resolution
    # (validated separately); run it there, then upscale the mask back to full res.
    scale = 700 / max(h, w)
    small = cv2.resize(bgr, (int(w*scale), int(h*scale)))
    sh, sw = small.shape[:2]
    rect = (int(sw*pad[0]), int(sh*pad[1]), int(sw*pad[2]), int(sh*pad[3]))
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)

    if seeds:
        # seeded mode: start everything as probable-background, mark the outer rect as
        # probable-foreground, then stamp in definite fg/bg rects on top. This constrains
        # grabCut far more than a bare rect when foreground/background colors are similar.
        mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
        pr_fg = seeds.get("pr_fg")
        if pr_fg:
            for (bx0,by0,bx1,by1) in pr_fg:
                mask[int(by0*scale):int(by1*scale), int(bx0*scale):int(bx1*scale)] = cv2.GC_PR_FGD
        else:
            x0,y0,x1,y1 = rect
            mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
        for (bx0,by0,bx1,by1) in seeds.get("fg", []):
            mask[int(by0*scale):int(by1*scale), int(bx0*scale):int(bx1*scale)] = cv2.GC_FGD
        for (bx0,by0,bx1,by1) in seeds.get("bg", []):
            mask[int(by0*scale):int(by1*scale), int(bx0*scale):int(bx1*scale)] = cv2.GC_BGD
        cv2.grabCut(small, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    else:
        mask = np.zeros((sh, sw), np.uint8)
        cv2.grabCut(small, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask==2)|(mask==0), 0, 255).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))

    # grabCut sometimes keeps stray patches of wall/floor that survive the rect alone
    # (not strongly enough contrasted for the color model to carve out). Those patches
    # are rarely solidly connected to the chair silhouette, so: erode to snap weak
    # bridges, keep only the largest blob (the chair), then dilate back to size.
    k = np.ones((7,7), np.uint8)
    eroded = cv2.erode(fg, k, iterations=3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        eroded = np.where(labels == largest, 255, 0).astype(np.uint8)
    fg = cv2.dilate(eroded, k, iterations=3)
    fg = cv2.bitwise_and(fg, cv2.morphologyEx(np.where((mask==2)|(mask==0),0,255).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9,9),np.uint8)))

    fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)
    fg = np.where(fg > 127, 255, 0).astype(np.uint8)
    return fg

def classify(bgr, fg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    is_accent = (H >= ACCENT_HUE[0]) & (H <= ACCENT_HUE[1]) & (S >= ACCENT_SAT_MIN) & (V >= ACCENT_VAL_MIN)
    is_body = (~is_accent) & (V < BODY_VAL_MAX)
    is_panel = (~is_accent) & (~is_body)

    zones = {}
    k = np.ones((3,3), np.uint8)
    for z, m in (("body", is_body), ("panel", is_panel), ("accent", is_accent)):
        raw = np.where(m & (fg>0), 255, 0).astype(np.uint8)
        # drop stray single/few-pixel flecks (JPEG noise, anti-aliased edges that happen to
        # land in the accent hue band) without eating the real thin accent stripe
        zones[z] = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    return zones

def soften(mask, sigma=2.5):
    return cv2.GaussianBlur(mask, (0,0), sigma)

def process(name, cfg):
    path = RAW / f"{name}.jpg"
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    fg = grabcut_fg(bgr, cfg["rect_pad"], cfg.get("seeds"))
    zones = classify(bgr, fg)

    armrest = np.zeros((h,w), np.uint8)
    for (x0,y0,x1,y1) in cfg["armrest_boxes"]:
        box_body = np.zeros((h,w), np.uint8)
        box_body[y0:y1, x0:x1] = zones["body"][y0:y1, x0:x1]
        armrest = np.maximum(armrest, box_body)
        # remove armrest pixels from the body zone so they don't double-recolor
        zones["body"][y0:y1, x0:x1] = np.where(box_body[y0:y1, x0:x1]>0, 0, zones["body"][y0:y1, x0:x1])

    # NOTE: saved as fully-opaque RGB (not RGBA) -- browsers premultiply/blend by the alpha
    # channel when drawing a PNG onto a canvas, which would silently zero out R/G/B wherever
    # alpha (armrest) was low. Two opaque images avoids that trap entirely.
    packed = np.zeros((h, w, 3), dtype=np.uint8)
    for i, z in enumerate(ZONE_ORDER):
        packed[:,:,i] = soften(zones[z])
    Image.fromarray(packed, "RGB").save(OUT / f"{name}_mask.png")

    arm_img = np.zeros((h, w, 3), dtype=np.uint8)
    arm_img[:,:,0] = soften(armrest)
    Image.fromarray(arm_img, "RGB").save(OUT / f"{name}_armrest.png")

    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    Image.fromarray(lum).convert("L").save(OUT / f"{name}_lum.jpg", quality=92)

    Image.fromarray(rgb).save(OUT / f"{name}_photo.jpg", quality=92)

    # debug overlay so results can be eyeballed before wiring into the viewer
    overlay = rgb.copy()
    tint = {"body": (255,60,60), "panel": (60,140,255), "accent": (60,255,140)}
    for z, color in tint.items():
        m = zones[z] > 80
        overlay[m] = (overlay[m]*0.35 + np.array(color)*0.65).astype(np.uint8)
    am = armrest > 80
    overlay[am] = (overlay[am]*0.35 + np.array([255,230,0])*0.65).astype(np.uint8)
    Image.fromarray(overlay).save(OUT / f"{name}_debug.png")

    has_armrest = bool(cfg["armrest_boxes"])
    return {"zones": ["body","panel","accent"] + (["armrest"] if has_armrest else []), "size":[w,h]}

if __name__ == "__main__":
    manifest = {}
    for name, cfg in SCENES.items():
        print("processing", name)
        manifest[name] = process(name, cfg)
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done ->", OUT)
