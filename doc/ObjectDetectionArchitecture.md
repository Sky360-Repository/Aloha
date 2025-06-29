## **Object Detection System Architecture**

### **1. Method for Object Detection**

#### **General Description**
The system follows a modular, multistage computer vision (CV) approach tailored for embedded deployment (e.g. Orange Pi 5+). It leverages diverse, complementary techniques for motion segmentation, blob detection, and region-of-interest analysis. Instead of depending on a single model or end-to-end neural network, it orchestrates lightweight, interpretable methods that reflect spatiotemporal events with high fidelity.

**Core methods used:**
- **GMM / ViBe**: Statistical background models to detect changes.
- **3-Frame Differences**: Intensity and gradient-based motion detection.
- **Dense Optical Flow (DOF)**: Tracks fast, smooth motion across frames.
- **DoG Blob Detection**: Highlights spatial saliency at multiple scales.
- **MHI**: Encodes recent motion as a temporal heatmap.
- **Codebook Models**: Compact pixel history alternative to GMM.
- **SIFT (and ORB optional)**: Matches persistent features during fast or distorted motion.

### **2. Detection Annotation (Criterion List)**

#### **General Description**
Each detected region (blob/point of interest) is annotated with a *Criterion List*—a structured set of flags or scores representing how and why a region was considered meaningful.

**Sample Criteria Include:**
- Presence in GMM/ViBe foreground
- Shape/texture persistence (DoG, Harris, SIFT)
- Temporal behavior (MHI score, slow/static blobs)
- Motion vectors and flow confidence (DOF)
- Appearance frequency or semantic relationship to POIs

This list serves both as an explainable output and as **input features for a downstream tinyML model**.

### **3. Method ↔ Criterion Mapping**

#### **How Methods Complement Each Other**
| Method                | Primary Strength                            | Weakness                       | Complementarity                                             |
|----------------------|----------------------------------------------|--------------------------------|-------------------------------------------------------------|
| GMM                  | Models multiple stable background modes      | Slow with static objects       | ViBe catches fast transitions, 3FD/DOF reveal ghosts       |
| ViBe                 | Fast background adaptation                   | Can forget valid structures    | GMM provides temporal depth and mode persistence           |
| 3-Frame Diff (RGB)   | Detects sudden motion edges                  | Poor with slow/stationary motion | Gradient+DOF fill gaps                                     |
| 3-Frame Gradient     | Texture-rich contour detection               | Sensitive to noise             | Filters for DoG/Harris, boosts structure in motion         |
| DOF                 | Tracks smooth motion                         | Sensitive to motion blur       | SIFT detects fast, non-linear keypoint changes             |
| DoG                 | Multi-scale spatial saliency                 | Not temporal                   | Paired with MHI and morphology to refine POIs              |
| Codebook Model       | Memory-efficient background modeling         | Lacks probabilistic weighting  | Fallback for low-power or cold-start background update     |
| SIFT                | Reliable feature matching under transformations | Fails in fisheye distortions   | Good for high-motion zones DOF can't explain              |
| MHI                 | Captures motion history and decay            | Not a detector per se          | Combines beautifully with POI saliency to score relevance  |

### **4. State Matching**

This layer uses **logic-based fusion rules** to decide when a region should trigger attention or alarm. For example:
```text
ATTENTION = GMM_foreground AND POI AND (MHI_score > threshold)
```

Such combinations allow:
- Identification of slow, static anomalies.
- Separation of fast motion from semantically meaningful motion.
- Filtering of false positives from shadows or flicker.

The **Criterion List** feeds these rules, enabling context-aware object detection with interpretable logic.

### **5. Detection Pipeline**

#### **Description of Steps**
1. **Blob Triggering**
   - Inputs: GMM, ViBe, 3-frame diffs, DOF
   - Output: initial motion mask

2. **Morphological Cleanup**
   - Operations: erosion, dilation, contour tracking
   - Output: cleaned blob list

3. **POI Detection**
   - Features: DoG (small & large σ), Harris, optional SIFT
   - Output: attention heatmaps

4. **Semantic Blob Association**
   - Match blobs ↔ POIs
   - Compute distances, feature overlap, recurrence
   - Update MHI and assign scores

5. **Model Updates (Asynchronous)**
   - Define background mask using MHI & semantic logic
   - Dispatch GMM, ViBe, Codebook, and MHI updates in parallel threads

6. **Output Construction**
   - For each blob: ROI data, modality snapshots, Criterion List
   - Passed into a small downstream ML model for labeling or event tagging

#### **Logic**
Each stage is gated by decisions from the previous one. Feedback from motion history (MHI) and semantic strength (POI) ensures that:
- Foreground blobs evolve only if meaningful.
- Models adapt only where it’s safe.
- Processing load is reserved for high-value regions.

### **6. Classical Approach vs End-to-End ML**

#### **Why Classical Makes Sense Here**
- **Data scarcity**: Labeling sufficient data for end-to-end training is impractical due to the specialized camera setup.
- **Optics sensitivity**: Classical methods adapt to distortion and artifacts without needing retraining.
- **Interpretability**: Each decision is explainable via clear criteria.
- **Modularity**: Easy to tune, extend, or disable components on constrained hardware.
- **Temporal handling**: Models like GMM, ViBe, and MHI inherently capture time dynamics that typical CNNs struggle to replicate.

In this setup, ML becomes an assistant—not a replacement—integrating only at the final semantic or classification layer.

