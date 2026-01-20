# Bézier Curves and Hermite Splines in Geometric Modeling

## Project Overview

This project was carried out as part of a **geometric modeling** course and focuses on the study, implementation, and analysis of **Bézier curves** and **Hermite splines**.  
The main objective is to understand the mathematical properties of these curves, their continuity, geometric behavior, and visual quality.

A comparative study with **Lagrange polynomial interpolation** is also conducted in order to highlight the strengths and limitations of each method.

---

## Theoretical Background

The project is based on:
- Hermite polynomials and their dual basis
- The relationship between Hermite splines and Bézier curves
- **C¹** and **C²** continuity conditions
- Curvature as a geometric quality indicator
- The impact of tangent selection on curve behavior

---

## Implementation

The implemented features include:
- Construction of **C¹ Hermite splines** with uniform parameterization
- Use of a tension parameter `c`
- Tangent computation at boundary and internal points
- Extension to **C² Hermite splines** ensuring second-derivative continuity
- Curvature computation and visualization
- Visual comparison between:
  - C¹ Hermite splines
  - C² Hermite splines
  - Lagrange interpolation

---

