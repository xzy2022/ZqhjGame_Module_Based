# Third-party notices

This archive contains an application assembled from the project source, the supplied trained model,
and the following third-party components. Original license notices are retained. This document does
not replace the licenses or assign a new license to the user's source, dataset samples or weights.

| Component | Version / source | License location in this archive |
|---|---|---|
| CPython | 3.13.5, python.org embeddable x64 distribution | `runtime/LICENSE.txt` |
| Ultralytics | 8.4.154, source retained verbatim | `.runtime/ultralytics-8.4.154/ultralytics-8.4.154.dist-info/licenses/LICENSE`, AGPL-3.0 |
| PyTorch / bundled CUDA libraries | 2.9.1+cu128, official PyTorch wheel | `runtime/Lib/site-packages/torch-2.9.1+cu128.dist-info`, plus notices distributed within `torch/` |
| torchvision | 0.24.1+cu128 | corresponding `.dist-info` directory |
| NumPy, SciPy, OpenCV, Pillow and remaining Python dependencies | exact versions in `requirements.txt` | original `LICENSE*`, `COPYING*`, `NOTICE*` files and `.dist-info` metadata under `runtime/Lib/site-packages/` |
| Microsoft Visual C++ Runtime | 14.44.35211.0, DLLs extracted from the Microsoft-signed x64 Redistributable | `licenses/Microsoft_VC_Runtime_LICENSE_en.rtf`, `Microsoft_VC_Runtime_LICENSE_zh.rtf`; provenance in `reports/msvc_runtime.json` |

Official sources:

- https://www.python.org/ftp/python/3.13.5/python-3.13.5-embed-amd64.zip
- https://download.pytorch.org/whl/cu128
- https://pypi.org/project/ultralytics/8.4.154/
- https://github.com/ultralytics/ultralytics
- https://aka.ms/vs/17/release/vc_redist.x64.exe

The application's installed dependency inventory is in `reports/dependency_check.json`.
Distribute the source and license files with the executable runtime, rather than redistributing only model binaries.
