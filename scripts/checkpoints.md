# Where to get published detector weights

All are single-GPU friendly. Vendor author code under `third_party/` and wrap
with `aigi_bench.detectors.external.TorchModuleDetector`.

| Method | Paper | Code |
|---|---|---|
| UniversalFakeDetect | Ojha et al., CVPR 2023 | github.com/WisconsinAIVision/UniversalFakeDetect |
| NPR | Tan et al., CVPR 2024 | github.com/chuangchuangtan/NPR-DeepfakeDetection |
| FreqNet | Tan et al., AAAI 2024 | github.com/chuangchuangtan/FreqNet-DeepfakeDetection |
| AIDE | Yan et al., 2024 | github.com/shilinyan99/AIDE |
| DRCT | Chen et al., ICML 2024 | github.com/beibuwandeluori/DRCT |
| DIRE | Wang et al., ICCV 2023 | github.com/ZhendongWang6/DIRE |
| C2P-CLIP | Tan et al., 2025 | github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection |
| SAFE | Li et al., 2025 | github.com/Ouxiang-Li/SAFE |

Verify each URL/checkpoint hash from the paper before use; repos move.

Datasets: GenImage, UniversalFakeDetect test sets, Synthbuster, Chameleon,
NTIRE 2026 challenge data, RAID (adversarial stress-test, evaluation only).
