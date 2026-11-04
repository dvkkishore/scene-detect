# Demonstrations and verification

The README visuals combine **actual model output** with a separately identified architecture diagram. This page records the inputs, commands, execution boundary, and redistribution basis. These are reproducibility checks, not an accuracy study.

## Recorded environment

Recorded on **4 November 2026**, using the existing Linux virtual environment and Python **3.10.12**. The implementation base was GitHub `main` at `fc24a73497593e7abd5fc2b9d5206d716e775c8a`. The three detection scripts and `pose_classifier.py` were unchanged for these runs.

| Dependency | Installed version |
| --- | --- |
| MediaPipe | `0.10.21` |
| NumPy | `1.26.4` |
| OpenCV Python + contrib distributions | Both `4.11.0.86` in the existing environment |
| Ultralytics | `8.3.156` |
| PyTorch | `2.7.0+cu126`; these inference runs used CPU |
| pyttsx3 | `2.98` |
| Pillow (visual layouts) | `11.2.1` |

The MediaPipe full and heavy pose model files were already installed. No model download was needed. YOLO used the repository's existing `yolov8n-pose.pt`, SHA-256:

```text
c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212
```

The focused [requirements_demo.txt](../requirements_demo.txt) records the direct demo dependency versions. It is not a lockfile or a clean-install certification. PyTorch and transitive dependency resolution depend on the platform. The original exported requirement files were retained; they were not installed into a fresh environment.

## Recorded results

Every output below is the original 512 × 512 canvas saved without retouching. The presentation layouts resize whole images, but the linked output PNGs retain the original pixels. JSON records contain input/output hashes and script hashes; the live records also contain the actual drawn strings and intercepted speech requests.

| Variant | Input | Observed result | Canvas | Record |
| --- | --- | --- | --- | --- |
| Image / MediaPipe | NASA portrait | `Person OK (Posture Flag Clear)` | [Portrait](assets/outputs/image-nasa-astronaut.png) | [JSON](assets/records/image-nasa-astronaut.json) |
| Image / MediaPipe | Portrait rotated 90° clockwise | `Person NOT OK (Posture Flag)` | [Rotation](assets/outputs/image-nasa-astronaut-rotated.png) | [JSON](assets/records/image-nasa-astronaut-rotated.json) |
| Image / MediaPipe | Generated white image | `No Person Detected` | [Blank](assets/outputs/image-blank.png) | [JSON](assets/records/image-blank.json) |
| Live / MediaPipe | NASA portrait, one-frame replay | `Person OK` | [Portrait](assets/outputs/mediapipe-live-portrait.png) | [JSON](assets/records/mediapipe-live-portrait.json) |
| Live / MediaPipe | Rotated portrait, one-frame replay | `Person NOT OK`; speech request `Person not okay!` | [Rotation](assets/outputs/mediapipe-live-rotated.png) | [JSON](assets/records/mediapipe-live-rotated.json) |
| Live / YOLO | NASA portrait, one-frame replay | `Insufficient Pose` | [Portrait](assets/outputs/yolo-portrait.png) | [JSON](assets/records/yolo-portrait.json) |
| Live / YOLO | NASA crew portrait, one-frame replay | Four `Person OK` labels; five people in input | [Crew](assets/outputs/yolo-crew.png) | [JSON](assets/records/yolo-crew.json) |

The rotated portrait is a deliberately transformed **orientation stress input**, not a photograph of a person lying down. It supplies no evidence about an emergency. The partially visible portrait also illustrates disagreement between variants. YOLO's `Insufficient Pose` text is clipped at the top of its original canvas; that rendering defect is preserved. The crew image demonstrates multiple poses and a missed person in the same frame.

### Reproduce the image CLI outputs

From the repository root, in a compatible environment:

```bash
python detection_with_images.py \
  --image docs/assets/inputs/nasa-astronaut.jpg \
  --output demo_frames/image-portrait.png

python detection_with_images.py \
  --image docs/assets/inputs/nasa-astronaut-rotated.jpg \
  --output demo_frames/image-rotated.png

python detection_with_images.py \
  --image docs/assets/inputs/blank.png \
  --output demo_frames/image-blank.png
```

The actual capture runs saved those files under `docs/assets/outputs/`; each [image JSON record](assets/records/image-nasa-astronaut.json) records its exact CLI command. Use `demo_frames/` for local reruns to avoid overwriting the checked-in evidence. Small inference differences across hardware or dependency versions may change labels, keypoints, or output hashes.

### Reproduce the live-script captures

[tools/capture_live_demo.py](../tools/capture_live_demo.py) runs the selected original script with `runpy`. It redirects `cv2.VideoCapture` to a **local file** and intercepts `imshow`, `waitKey`, `destroyAllWindows`, and `pyttsx3.init`. The estimator, posture rules, and drawing operations run as written. It records `putText` calls while forwarding them to OpenCV. No model results are mocked.

The tool exports the last displayed canvas and fails if none was produced. For the bundled single-frame images, this is the only frame. Speech requests are recorded but never spoken, and no preview window opens. The tool is needed because the original live scripts require camera/display/audio I/O at module scope; a separate harness makes documentation capture possible without changing their behavior.

```bash
python tools/capture_live_demo.py --variant mediapipe \
  --input docs/assets/inputs/nasa-astronaut.jpg \
  --output demo_frames/mediapipe-portrait.png \
  --record demo_frames/mediapipe-portrait.json

python tools/capture_live_demo.py --variant mediapipe \
  --input docs/assets/inputs/nasa-astronaut-rotated.jpg \
  --output demo_frames/mediapipe-rotated.png \
  --record demo_frames/mediapipe-rotated.json

python tools/capture_live_demo.py --variant yolo \
  --input docs/assets/inputs/nasa-astronaut.jpg \
  --output demo_frames/yolo-portrait.png \
  --record demo_frames/yolo-portrait.json

python tools/capture_live_demo.py --variant yolo \
  --input docs/assets/inputs/nasa-sts93-crew.jpg \
  --output demo_frames/yolo-crew.png \
  --record demo_frames/yolo-crew.json
```

The recorded live commands used the same arguments with output/record paths under `docs/assets/outputs/` and `docs/assets/records/`. The verification environment used `env/bin/python`, with `MPLCONFIGDIR=/tmp/scene-mpl` and `YOLO_CONFIG_DIR=/tmp/scene-yolo` to keep library settings in writable temporary directories.

To rebuild just the README layout images and architecture SVG from the saved canvases:

```bash
python tools/build_doc_visuals.py
```

That layout tool requires Pillow and the DejaVu Sans fonts at `/usr/share/fonts/truetype/dejavu/`. It performs no inference. Separate desktop/mobile layouts let the README stack its input/output panels on narrow displays.

## Verification summary

| Check | What actually ran | Result / boundary |
| --- | --- | --- |
| Existing tests | `env/bin/python -m unittest discover -s tests -v` | All six original tests passed |
| Capture tests | Same discovery command, including `test_demo_capture.py` | Three additional I/O tests passed; **9 passed total** |
| Lightweight environment | `python -m unittest discover -s tests -v` with system Python/NumPy | Six passed, three capture tests explicitly skipped because pyttsx3 was absent |
| CLI arguments | `env/bin/python detection_with_images.py --help` | Passed; no ML imports needed for help |
| Syntax | `python -m compileall -q detection_with_images.py detection_with_live_mediapipe.py detection_with_live_yolo.py pose_classifier.py tools tests` | Passed |
| Real inference | Three direct image-CLI runs and four live-script local-file replays | Canvases and records linked above |
| Original camera entry points | Both live scripts launched directly with the existing environment | Both exited after OpenCV could not open camera `0`; no `/dev/video*` device was available |
| Speech startup | `pyttsx3.init()` and original live-script startup | Initialization succeeded; audible playback was not tested |
| Model data | Existing MediaPipe files and included YOLO weights | Loaded; first-use downloads not tested |
| Installation | Existing environment only | No fresh Linux install or Windows execution verified |

The capture I/O tests exercise file replay, canvas export, label recording, speech interception, failure with no displayed frame, and a missing input. They do not replace model outputs or test detection accuracy. The existing CI's NumPy-only environment skips these optional I/O tests while retaining the six original checks.

### Rendered documentation review

All four Markdown documents were rendered locally with Markdown2 and reviewed with Playwright using Brave Chromium at **1280 × 900** and **390 × 844** viewports. All embedded images and the SVG loaded; there was no page-level horizontal overflow at either width. Wide tables and code blocks scroll within their own containers. The README selected the stacked mobile hero at the narrow viewport. Full-page screenshots were generated for inspection; the preview used GitHub-style CSS rather than GitHub's server renderer.

Relative file links, local heading anchors, execution-record hashes, asset sizes, and the contributor names were checked. The 21 documentation assets total approximately **1.04 MiB**; each is under **400 KiB**. The diagram is self-contained SVG, with no externally loaded fonts or images. Supporting material stays in the two technical documents and the existing evaluation protocol rather than duplicating full explanations in the README.

Hardware camera acquisition, real OpenCV windows, sustained video tracking, audible speech, alert reset sequences, performance, and a labeled real-world evaluation remain outside the executed checks.

## Asset provenance

### Added photographic inputs

Source pages and usage terms were checked on **4 November 2026**. These photos are used to explain model behavior in an engineering demonstration; neither NASA nor any depicted person endorses the project.

| Added file | Source and credit | Redistribution basis | Preparation |
| --- | --- | --- | --- |
| [nasa-astronaut.jpg](assets/inputs/nasa-astronaut.jpg) | NASA portrait of Eileen Collins, distributed as scikit-image's `astronaut` sample; [source file at v0.25.2](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/astronaut.png) | [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut) identifies it as public domain with no known copyright restrictions | Original 512 × 512 PNG converted to RGB JPEG, Pillow quality `88`, `optimize=True`; dimensions retained, metadata omitted |
| [nasa-astronaut-rotated.jpg](assets/inputs/nasa-astronaut-rotated.jpg) | Same NASA/scikit-image portrait | Same public-domain source | Original PNG rotated 90° clockwise with OpenCV, then saved with the same JPEG settings; not a new photographic scenario |
| [nasa-sts93-crew.jpg](assets/inputs/nasa-sts93-crew.jpg) | NASA, STS-93 crew; [mission page and credit](https://www.nasa.gov/mission/sts-93/), [downloaded photograph](https://www.nasa.gov/wp-content/uploads/2023/04/sts-93-crew.jpg) | [NASA media guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/) permit factual educational/informational use with credit and without implying endorsement; the mission page credits NASA | Original 2584 × 2018 JPEG resized proportionally to 960 × 750 with Pillow `thumbnail`, then saved as RGB JPEG at quality `88`, `optimize=True`; metadata omitted |

The crew image is used under NASA's media-use guidance, not a newly assigned project license. Insignia appearing within the photograph are incidental parts of that photo, not a project logo. Original crew download SHA-256: `5745f26c3c4b717a0ffedbf60e4e98aac050d38c21f5f5507c48f39ecd6d514a`. Input and output hashes for the files actually used in inference are in the execution records.

### Created assets

| Files | Origin and treatment |
| --- | --- |
| [blank.png](assets/inputs/blank.png) | Independently generated 512 × 512 RGB image, every channel set to 255; no third-party material |
| `assets/outputs/*.png` | Seven canvases captured from repository inference on the documented inputs; retain the source-image credits above |
| `scene-detect-demo.png`, `scene-detect-demo-mobile.png` | Layouts combining the NASA crew photo and actual YOLO canvas, with editorial captions; generated by `tools/build_doc_visuals.py` |
| [architecture.svg](assets/architecture.svg) | Original vector diagram written for this documentation; **conceptual**, not model output |
| `assets/records/*.json` | Execution records created during the capture runs; contain relative paths, hashes, output strings, and muted speech requests |

No image-generation model was used to create inference examples. No license for the existing repository has been added or changed.

### Existing samples and supplied references

The three original folders contain **40 photographs**: 24 in `Person OK/`, 10 in `Person not OK/`, and 6 in `Nobody/`. Some filenames suggest Unsplash or Pexels creators, but filenames alone do not establish source identity or redistribution permission. No source/permission manifest or top-level license was found, and the checked EXIF copyright fields did not provide rights records. All originals remain untouched; none is newly embedded in the documentation.

The supplied `Person OK/9.jpeg` has no verified source/permission record. The two supplied skeleton screenshots lack a reproducible input/command/variant record for the current implementation. They were used as presentation references, not published as newly verified results. The supplied `OTH_Poster.pdf` was read as historical project context; its visual assets and stronger performance/privacy claims were not copied into the repository. The contributor list remains the original six-person list, without invented responsibilities.

Before featuring an original photo, record its exact source URL, author, applicable terms, and permitted use. Before featuring an old screenshot, also identify its source input, code version, and execution path.
