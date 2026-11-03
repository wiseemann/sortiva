# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **School Community**: Vocational high school (SMK) students, teachers, and school staff interacting with the smart sorting bin kiosk or web camera interface for real-time waste disposal guidance.
- **Adiwiyata Committee & Administrators**: Environmental coordinators and school staff monitoring campus waste generation, segregation compliance, category volume distributions, and environmental audit metrics.

## Product Purpose

Provide an AI-powered waste classification and segregation guidance system designed for school environments under the Indonesian Adiwiyata eco-school framework. Success means students and staff segregate waste with zero confusion into designated school sorting compartments (Organik, Anorganik/Daur Ulang, and Residu/B3), backed by transparent campus waste analytics and verifiable database logging.

## Positioning

An integrated educational and physical smart-bin vision system built directly for school sustainability programs. Unlike generic municipal or industrial waste portals, it pairs real-time YOLOv8m computer vision with physical 3-compartment sorting mechanics (top-rotor indexing with stationary storage) and school-level waste literacy.

## Operating Context

- Deployed as an interactive smart bin kiosk, live webcam sorting station, or web dashboard in school hallways, canteens, and waste collection points.
- Operates under ambient indoor and outdoor lighting with potential clutter or multi-item submissions.
- Built to run efficiently on standard desktop/laptop CPU hardware without requiring dedicated external GPUs or cloud roundtrips.
- Logs all detections and image reports to a local SQLite database (`waste_data.db`).

## Capabilities and Constraints

- **Confirmed Capabilities**:
  - Real-time live camera feed detection with class predictions, confidence ratings, and compartment guidance.
  - Drag-and-drop photo upload analysis with whole-image and overlapping 4-tile scan (NMS merged) to catch small items in cluttered piles.
  - 8-class YOLOv8m detection model (`plastic`, `metal`, `glass`, `e_waste`, `medical_waste`, `paper`, `cardboard`, `organic_waste`).
  - Nature categorization: Biodegradable, Non-Biodegradable, Hazardous/Residu.
  - Historical analytics dashboard (`pages/dashboard.py`) showing total detections, class breakdown, confidence trends, source splits, and CSV export.
  - 3D interactive hardware mockup (`AI Smart Waste Sorting Bin - Adiwiyata 3D Mockup.html`) illustrating top-rotor mechanism and 3-compartment bin architecture.
- **Technical Constraints**:
  - Python 3.10 / 3.11 environment constraint (pinned PyTorch 2.1.2 CPU dependencies).
  - Robust multi-platform webcam backend handling (Windows DirectShow/MSMF and Linux V4L2).
- **Open Decisions**:
  - Transition and finalize the frontend UI mapping from the 8 model classes into the 3 physical Adiwiyata school compartments (Organik: Hijau, Daur Ulang / Anorganik: Kuning/Biru, Residu / B3: Merah/Abu-abu).

## Brand Commitments

- **Official Name**: **SORTIVA** (Smart Optical Recognition & Trash Identifier for Adiwiyata).
- **Sub-title & Tagline**: *"The Stembayo Eco-Sort Platform"* — SMK Negeri 2 Depok Sleman.
- **AI Vision Engine**: **StemVision Engine** (YOLOv8m 8-class computer vision module).
- **Mascot & Virtual Assistant**: **MAS BAYO** — Friendly, educational, and smart companion assisting students, teachers, and school staff with real-time waste segregation.
- **Brand Voice & Personality**: Friendly, educational, sharp, and eco-tech focused. Integrates school pride (Stembayo) with institutional environmental responsibility (Adiwiyata Mandiri).
- **Deprecations**: Fully retired legacy upstream Indian government / Swachh Bharat portal elements (tricolour strips, Ashoka blue headers, gazette reference numbers).

## Evidence on Hand

- **Model Weights**: `models/best.pt` (trained YOLOv8m 8-class model).
- **Core Application**: Streamlit main interface (`app.py`) and analytics dashboard (`pages/dashboard.py`).
- **Database & Data**: SQLite database (`waste_data.db`) storing logged detections and reports; database controller (`database.py`).
- **Hardware Mockup**: `AI Smart Waste Sorting Bin - Adiwiyata 3D Mockup.html` specifying 3-sector stationary bin and rotor mechanics.
- **Configuration**: Model configuration files (`config/data.yaml`, `config/data_yolov8m.yaml`).

## Product Principles

- **Instant Disposal Clarity**: At the point of disposal, bin guidance must be readable at a glance in under a second with distinct color-coding, iconography, and high visual contrast.
- **Educational Literacy**: Every scan should teach the user *why* an item belongs in a specific bin and its ecological nature under the Adiwiyata program.
- **Audit-Grade Traceability**: All classification events, confidence intervals, and timestamps remain persistent and exportable for campus sustainability reports.
- **Lightweight Reliability**: The UI and inference pipeline must remain responsive and dependable on everyday school computers without cloud dependencies.

## Accessibility & Inclusion

- Multi-modal feedback: High-contrast color indicators paired with clear text labels, bin icons, and large readable typography to ensure effortless use by students with color-vision deficiencies or under varied school lighting.

