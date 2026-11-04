"""Build documentation layouts from captured output; never generate detections.

Requires Pillow. Run from the repository root after reproducing the captures.
The PNG layouts resize entire canvases without cropping or repainting them.
The SVG is a conceptual architecture diagram, not an inference result.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parents[1] / 'docs' / 'assets'
FONT = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    return ImageFont.truetype(str(FONT / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)


def fit(image, size):
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    return result


def hero(mobile=False):
    width, height = (640, 1110) if mobile else (1200, 670)
    im = Image.new('RGB', (width, height), '#101e2e')
    d = ImageDraw.Draw(im)
    d.text((32, 26), 'SCENE-DETECT / CAPTURED OUTPUT', font=font(18, True), fill='#67d8cf')
    d.text((32, 66), 'From pixels to posture cues', font=font(30 if mobile else 38, True), fill='white')
    d.text((32, 120), 'YOLOv8-Pose · unchanged live renderer', font=font(20), fill='#ccd8e7')
    photo = Image.open(ASSETS / 'inputs/nasa-sts93-crew.jpg').convert('RGB')
    output = Image.open(ASSETS / 'outputs/yolo-crew.png').convert('RGB')
    panels = [(32, 178, 576, 370, 'INPUT / NASA STS-93 CREW', photo),
              (32, 604, 576, 370, 'OUTPUT / FOUR DETECTED POSES', output)] if mobile else [
              (32, 186, 552, 396, 'INPUT / NASA STS-93 CREW', photo),
              (616, 186, 552, 396, 'OUTPUT / FOUR DETECTED POSES', output)]
    for x, y, w, h, label, source in panels:
        d.text((x, y-32), label, font=font(17, True), fill='#ccd8e7')
        d.rounded_rectangle((x, y, x+w, y+h), radius=12, fill='white')
        resized = fit(source, (w-16, h-16))
        im.paste(resized, (x+(w-resized.width)//2, y+(h-resized.height)//2))
    y = 1010 if mobile else 610
    d.text((32, y), 'Five people in the photo; four detected.', font=font(20, True), fill='white')
    d.text((32, y+32), 'A reproducible frame, not a performance benchmark.', font=font(18), fill='#ccd8e7')
    im.save(ASSETS / ('scene-detect-demo-mobile.png' if mobile else 'scene-detect-demo.png'), optimize=True)


def architecture():
    # Vertical layout keeps the diagram legible when embedded in a narrow page.
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="796" viewBox="0 0 720 796" role="img" aria-labelledby="title desc">
<title id="title">Scene-detect data flow and variant rules</title>
<desc id="desc">RGB input enters a pose estimator. MediaPipe returns one pose with 33 landmarks; YOLO returns multiple poses with 17 points each. Variant-specific rules assign posture labels, then the scripts draw skeletons and optionally request speech. RGB processing occurs before the skeleton display.</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#526b80"/></marker></defs>
<rect width="720" height="796" rx="18" fill="#f2f6fa"/>
<g font-family="DejaVu Sans,Arial,sans-serif" fill="#172c40">
<text x="32" y="44" font-size="16" font-weight="bold" fill="#326e77">SYSTEM OVERVIEW / CONCEPTUAL DIAGRAM</text>
<rect x="32" y="68" width="656" height="90" rx="12" fill="#172c40"/>
<text x="56" y="104" font-size="24" fill="white" font-weight="bold">01  Image or camera frame</text>
<text x="56" y="135" font-size="19" fill="#dae5ee">Raw pixels are processed in memory.</text>
<path d="M360 158V195" stroke="#526b80" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="32" y="202" width="656" height="116" rx="12" fill="white" stroke="#c7d4df"/>
<text x="56" y="238" font-size="24" font-weight="bold">02  Pose estimation</text>
<text x="56" y="271" font-size="20">MediaPipe: one pose / 33 landmarks</text>
<text x="56" y="300" font-size="20">YOLOv8-Pose: multiple poses / 17 points each</text>
<path d="M360 318V355" stroke="#526b80" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="32" y="362" width="656" height="196" rx="12" fill="white" stroke="#c7d4df"/>
<text x="56" y="399" font-size="24" font-weight="bold">03  Variant-specific posture rules</text>
<text x="56" y="436" font-size="20">Image: wide bounds OR torso tilt &gt; 30°</text>
<text x="56" y="472" font-size="20">Live MediaPipe: 2 of 3 cues, including relative z</text>
<text x="56" y="508" font-size="20">Live YOLO: wide bounds AND torso tilt &gt; 30°</text>
<text x="56" y="540" font-size="17" fill="#526b80">YOLO requires visible shoulders and hips; no depth cue.</text>
<path d="M360 558V595" stroke="#526b80" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="32" y="602" width="656" height="118" rx="12" fill="#dcefeb" stroke="#a5cdc5"/>
<text x="56" y="640" font-size="24" font-weight="bold">04  Skeleton canvas + posture label</text>
<text x="56" y="675" font-size="20">512 × 512 display; optional image export</text>
<text x="56" y="704" font-size="20">Live scripts request speech when a flag first appears.</text>
<text x="32" y="761" font-size="19" fill="#526b80">Posture cues describe geometry, not a person's health.</text>
</g></svg>'''
    (ASSETS / 'architecture.svg').write_text(svg)


if __name__ == '__main__':
    hero()
    hero(mobile=True)
    architecture()
