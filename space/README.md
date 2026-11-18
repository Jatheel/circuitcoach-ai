---
title: CircuitCoach AI
emoji: 🔌
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: agpl-3.0
models:
  - JatheelMHD/circuitcoach-yolo26n
short_description: Identify electronics parts in a photo, in your browser
---

# CircuitCoach AI — web demo

Choose or take a photo of electronics and robotics parts. A YOLO26n model finds each part,
draws a box around it and shows a short knowledge card: what it does, key facts, cautions,
what the camera cannot tell, and sources.

- **Runs in your browser** with ONNX Runtime Web — the photo never leaves your device
- Model: [JatheelMHD/circuitcoach-yolo26n](https://huggingface.co/JatheelMHD/circuitcoach-yolo26n)
  (test mAP50 0.958 on the held-out ElectroCom61 split; in-lab photos)
- 61 component classes; knowledge cards for the most common learning-kit parts

**Safety:** for low-voltage (5 V or less) learning circuits. Not a safety check. Never use
on mains wiring.

**Credits:** ElectroCom61 dataset by Sayeedi et al., *Data in Brief* 59 (2025) 111331,
CC BY 4.0. Model and app: AGPL-3.0.
