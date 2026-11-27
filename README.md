# my_first_project
Tools and scripts for Point Cloud data analysis and HBIM integration.
## Project Objective
This repository brings together scripts and tools for point cloud processing, surface reconstruction, and feature-preserving mesh refinement—focused on complex architectural heritage elements such as **muqarnas**—as a foundation toward a broader automated Scan-to-HBIM workflow:
1. **Noise Cleaning & Reconstruction:** Process point clouds and generate triangle meshes while highlighting low-density / inferred regions where scan data is missing.
2. **Feature-Preserving Refinement & Traceability:** Smooth surface noise while preserving sharp edges and architectural details, with displacement tracking (`deviation` maps and JSON reports) for HBIM documentation.
3. **Future Direction (HBIM & AI):** Explore element segmentation/classification, missing-part completion, and parametric/IFC workflows in later stages.
   
## Roadmap
* [x] Resource-monitored Poisson surface reconstruction with density-based gap highlighting
* [x] Feature-preserving bilateral mesh refinement and geometric deviation reporting
* [ ] Point cloud / mesh segmentation and classification
* [ ] Missing-part completion for complex heritage geometry
* [ ] Parametric modeling and IFC integration
