# human_visualization

Visualization of human slices

# Interactive Volume Slicer (ModernGL) — MPR Plane UI + Heap/Brush Reveal + 3D Gizmo

Demo videos:
[![MPR Plane UI Demo](https://img.youtube.com/vi/VDaAYFwiu9o/hqdefault.jpg)](https://youtu.be/VDaAYFwiu9o)

[![3D Gizmo + View Cube Demo](https://img.youtube.com/vi/C_Gu3309b28/hqdefault.jpg)](https://youtu.be/C_Gu3309b28)

[![Heap / Brush Reveal Demo](https://img.youtube.com/vi/5wpkK5IAVzM/hqdefault.jpg)](https://youtu.be/5wpkK5IAVzM)

This project turns a stack of 2D medical slices into an interactive **3D volume viewer** with:

- **MPR plane slicing** (free-rotating plane through the volume)
- **Heap/brush reveal** (mouse-controlled depth reveal across a texture array)
- A **3D gizmo / view cube** (click X/Y/Z for axis planes, click 1–6 faces to snap camera + plane)
- Smooth navigation: rotate / pan / zoom / move plane through the volume

The volume data used in my workflow comes from the **Visible Human Project** PNG slices (abdomen/fullbody stacks), but the pipeline works for any consistent slice folder.

---

## Features

### 1) MPR Plane UI (full-screen slice)

- A single plane (center + orientation + scale) samples the 3D volume.
- The plane can rotate freely (yaw/pitch) and pan in its own U/V directions.
- Sampling blends between adjacent Z layers for smooth reslicing.

### 2) Heap / Brush Reveal (texture-array compositing)

- Loads N slices into a `sampler2DArray`.
- Mouse position sets a circular “brush”.
- Brush center shows _deeper_ layers; outside stays at the surface layer.
- Optional blending between layers vs. discrete layer stepping.

### 3) 3D Gizmo / View Cube

- Top-right 3D cube shows the volume bounds.
- Plane is rendered as a **3D slab** (extruded plane) rather than a flat “compressed texture”.
- Click:
  - **X / Y / Z** → snap the slicing plane normal to axis-aligned orientations
  - **1–6** → snap to cube faces (±X ±Y ±Z), like Unity view cube
  - Cube faces also support direct snapping by clicking the cube
