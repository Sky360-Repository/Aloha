# Gaussian Mixture of Models (GMM) for Background Subtraction

### Overview

This class implements a **Gaussian Mixture Model (GMM)** per pixel for background subtraction, suitable for video or time-lapse image streams.

### Initialization: `__init__()`

* **Inputs**:
  * `img_size`: Tuple of (Height, Width, Channels)
  * `nbr_gaussians`: Number of Gaussians per pixel (default 7)
  * `learning_factor`: Update rate per frame (e.g., 0.05 means slow learning)

* **Key Parameters**:
  * `self.mean`, `self.std`, `self.weight`: Shape = `(H, W, C, G)` or `(H, W, G)`
  * `initial_std`: Starting variance per Gaussian (very low, assuming little noise)
  * `nbr_curr_backgrounds`: Number of top Gaussians considered as valid background
  * `executor`: Thread pool with 1 worker for asynchronous background updates

* **Diagnostics**:
  * Prints estimated number of frames required to integrate new background pixels (ghosts)

Notes:
- This implementation avoids bounding mean and std, allowing adaptation to extreme pixel values (dark/saturated regions).
- Initialization isn't a performance bottleneck, but it mainly influences convergence speed.

### Background Update: `update(image, background_mask, std_factor)`

* Runs non-blocking update using `ThreadPoolExecutor`
* Skips if the previous frame is still processing (ensures thread safety)
* Submits `_threaded_update()` which performs the actual background model update

### GMM Update Logic: `_threaded_update()`

* Converts image to `[-0.5, 0.5]` float range
* Matches pixels to closest Gaussian (Mahalanobis distance)
* Updates matched Gaussians:

* Mean, variance, and weight using exponential moving average
* Replaces unmatched Gaussians (new foreground objects) with weak components
* Normalizes weights and sorts Gaussians by weight (descending)

### Foreground Detection: `get_difference_mask(image, std_factor)`

* Calculates Mahalanobis distance to top `N` Gaussians
* If no match below `std_factor`, pixel is considered **foreground**
* Returns binary mask (0 = background, 255 = foreground)

### Background Generation: `get_background()`

* Reconstructs background from the most dominant Gaussian (index 0)
* Converts back to `uint8` image in `[0, 255]` range

### Update Convergence: `get_update_rate(target_weight=0.7)`

* Estimates how many frames are needed to reach `target_weight` in a Gaussian component

* Formula:
  ```python
  n = ceil(log(1 - target_weight) / log(1 - alpha))
  ```

* Used to understand how long a "ghost" or new object takes to be absorbed into the background

### **Final Thoughts**
This approach avoids the costly **Expectation-Maximization (EM) algorithm** but achieves the **core GMM functionality efficiently**.
1. **Weight Update Stability** ? Ensure weights don�t fluctuate too fast or saturate early.
2. **Standard Deviation Evolution** - Monitor how `std` evolves over time if it collapses, variance handling may need adjustment.
3. **Score Sensitivity** - Confirm that `std_coefficient` tuning gives **clear separation** between Gaussians.
4. **Adapts fully to n-dimensional images**
5. **Efficient vectorized updates (avoids per-pixel explicit loops)**
6. **Memory-friendly optimization (no unnecessary reshaping or redundancies)**


## Comparison with Zivkovic's Adaptive GMM (MOG2)

This implementation shares the core idea of **per-pixel mixture of Gaussians**, but makes **deliberate choices for speed, control, and specific application needs**.

### Design Differences for Speed
* **No dynamic adjustment of number of Gaussians**

  * Zivkovic's method adapts the number of Gaussians per pixel over time based on data complexity.
  * This application fixes the number of Gaussians (default 7), avoiding per-pixel management logic **faster memory access and computation**.

* **Threaded asynchronous updates**

  * Zivkovic's method runs synchronously; frame processing time increases with scene complexity.
  * This application uses `ThreadPoolExecutor` to offload model update asynchronously **better frame rate stability**, especially at high FPS (e.g., 30 FPS).

* **Avoids per-frame sorting if not needed**
  * This model only sorts Gaussians **after normalization** and not during matching and saves compute cycles.

### Functional Differences
* **Explicit control over match masks and background integration**
  * This application supports an optional `background_mask` input to explicitly define where background learning is allowed (e.g., for controlled regions like the sky or static infrastructure).

* **Ghost integration time estimation**
  * `get_update_rate()` method allows predicting how many frames are required for a newly-exposed background to be integrated (e.g., ghosts).
  * Zivkovic's method doesn't have such introspection exposed - convergence time is implicit and variable.

* **Minimal variance enforcement**
  * Small fixed `min_std` to ensure numerical stability and consistent pixel matching even in extremely static scenes (like astronomy).
  * Zivkovic's method uses internal heuristics; may result in slow adaptation in noise-free but dynamic-lighting conditions.

### Simplifications / Assumptions

* **No shadow detection or color variance modeling**
  * Zivkovic's method  includes shadow detection logic and pixel color variance for better separation.
  * This application stripped down for pure speed and simplicity. Also because our application needs specific handling of shadows and highlights.

* **No background/foreground history tracking**
  * Zivkovic's method tracks persistence of background/foreground to handle flickering pixels.
  * This application works on **single-frame statistical update**, relying on spatial and temporal consistency of updates.

### Application-Driven Design Tradeoffs
* Built for **low-latency**, **high-frame-rate**, **always-on systems** like:
  * Sky cameras (e.g., all-weather fisheye QHY183c)
  * Environmental anomaly detection (lightning, meteors, clouds)

* Emphasis on **predictability and stability** over full automation:
  * This application *choose* learning rate instead of relying on adaptive alpha
  * This application *limits where* model updates are allowed with masks

### ? Summary Table

| Feature                      | Zivkovic MOG2        | This GMM Implementation         |
| ---------------------------- | -------------------- |---------------------------------|
| Adaptive Gaussian Count      | Yes                  | Fixed (faster)                  |
| Adaptive Learning Rate (`a`) | Yes (per-pixel)      | Fixed, user-defined             |
| Shadow Detection             | Yes                  | No                              |
| Multithreaded                | No                   | Yes (non-blocking)              |
| Background Control Mask      | No                   | Yes                             |
| Update Rate Estimation       | No                   | Yes                             |
| Background Pixel Replacement | Implicit (adaptive)  | Explicit logic                  |
| Use Case                     | General surveillance | High-throughput, custom systems |


## Pros and Cons of Gaussian Mixture Models (GMMs)

### Pros

* **Multiple Backgrounds per Pixel**
  * GMMs maintain a **mixture of Gaussians**, allowing a pixel to model multiple background states (e.g., swaying trees, flickering lights, sky/cloud transitions).
  * Great for environments with **intermittent occlusions or cyclic patterns**.

* **Fast Adaptation with Controlled Learning**
  * With a fixed `learning_factor`, GMMs can **gradually absorb changes** like lighting transitions or new static objects.
  * Ideal for **real-time systems** due to its low computational overhead (especially with fixed number of Gaussians and async processing).

* **No Need for Temporal Buffering**
  * Unlike frame-differencing or optical flow, GMMs **do not require multiple past frames** in memory - only per-pixel statistics.
  * Efficient for embedded or memory-constrained devices.

* **Pixel-Level Probabilistic Modeling**
  * Each pixel is treated independently, making the method **robust to localized motion** (e.g., a bird flying through the sky).

* **Custom Masking Support**
  * By using an external `background_mask`, you can **constrain where learning happens**, useful for long-term monitoring where ground objects are irrelevant (e.g., sky cameras).

### Cons

* **Ghost Artifacts (False Foregrounds)**
  * When an object leaves the scene, its outline (ghost) remains in the background model temporarily.
  * These need to be **externally managed**, or explicitly masked to prevent learning too early.
  * `get_update_rate()` helps **estimate ghost persistence time** based on learning rate.

* **No Semantic Awareness**
  * GMMs model pixels **independently**, so they don't understand what an "object" is (e.g., human, car, cloud).
  * Complex motion or contextual understanding requires **post-processing or external classifiers**.

* **Static Foreground Limitations**
  * If an object becomes stationary and `alpha` is high enough, it can be absorbed into the background (especially if no mask is used).
  * This makes **long-term foreground tracking harder** without external logic.

* **Noisy in Low-Light or High-Dynamic-Range Scenes**
  * If input noise is significant, GMMs may require **tuned `std_factor` thresholds**, or will become unstable without `min_std`.

## GMM vs Other Background Subtraction/Object Detection Methods

| Method                                       | Pros                                                | Cons                                                                            |
| -------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------- |
| **GMM** (this method)                        | Fast, handles multi-modal backgrounds, memory-light | Requires ghost handling, no semantic awareness                                  |
| **Frame Differencing**                       | Very fast, trivial to implement                     | Only detects motion, very sensitive to noise                                    |
| **Optical Flow**                             | Captures fine-grained motion                        | High computational cost, requires frame buffers                                 |
| **Deep Learning (e.g., YOLO/Mask R-CNN)**    | Semantic detection, object-level understanding      | Heavy compute, needs labeled data, less responsive to subtle background changes |
| **Codebook/ViBe/Pixel-Based Non-parametric** | Adaptive, good at preserving detail                 | Can be memory-hungry or harder to tune for stationary scenes                    |

## When to Use GMM

GMMs are **ideal for background subtraction in high-frame-rate, resource-efficient systems** where:

* Precise object identity isn't needed
* The background may have multiple stable states
* Fast response and control over learning are more important than semantic accuracy

## Complementary Methods to Enhance GMM Background Subtraction

While GMM is powerful for long-term background modeling, it benefits significantly from pairing with additional techniques to improve motion detection, ghost suppression, and robustness to dynamic scenes.

### Temporal Differencing (3-Frame Difference)

* **What**: Subtracts consecutive frames: `|frame_t - frame_t-1|` and `|frame_t-1 - frame_t-2|`
* **Use Case**: Highlights **moving objects**, suppresses static noise and shadows.
* **Complement to GMM**:
  * Helps isolate **truly moving objects** that GMM might misclassify (e.g., ghosts).
  * Low-cost and suitable for rapid transient detection.

### Single-Gaussian Background Model

* **What**: Per-pixel background with **one mean and variance**, updated quickly with a high learning rate.
* **Use Case**: Extremely responsive to scene changes - fast learning.
* **Complement to GMM**:
 * Can be used to **quickly absorb static backgrounds** (e.g., ghost regions), and contrast against the more conservative GMM.
 * Good for **short-term consistency checks** or highlighting objects that GMM is too slow to integrate.

### Optical Flow

* **What**: Estimates pixel-wise motion vectors between frames.
* **Use Case**: Detects **motion even with constant brightness**, great for slow-moving or intensity-invariant changes (e.g., fog, drifting objects).
* **Complement to GMM**:
 * Detects motion even when the object is similar to background in color.
 * Helps reject artifacts from noise or shadows.
 * Good for **post-GMM motion validation** or **trajectory estimation**.

### Gradient-Based Temporal Difference with Mode

* **What**: Uses **gradient orientation or magnitude** from three consecutive frames and takes the **mode** across time.
* **Use Case**: Detects motion that is **intensity-invariant** (e.g., silhouette in twilight, subtle smoke).
* **Complement to GMM**:
 * Extremely robust to illumination changes.
 * Useful for detecting subtle changes GMM might miss (e.g., fog boundaries, ghost removal).
 * More resilient than raw intensity difference.

### Difference of Gaussians (DoG)

* **What**: Blurs image at two scales and subtracts: `DoG = G(s1) - G(s2)`
* **Use Case**: Common in **astronomy and blob detection** (stars, meteor trails).
* **Complement to GMM**:
 * Helps **enhance point-like objects** (e.g., stars, planes, drones) over noisy background.
 * Especially good for **sky observations** and detecting local luminance peaks.

### Gabor Filters (Directional Frequency Detection)

* **What**: Convolution with sinusoidal wavelets oriented in various directions and frequencies.
* **Use Case**: Detects **texture-like patterns** (e.g., airplane smoke trails, contrails).
* **Complement to GMM**:
 * Effective at **spotting structured motion** not visible in frame differences (e.g., elongated smoke lines).
 * Can be used post-GMM to **classify foreground regions**.

### Deep Learning-Based Foreground Detectors (Optional)

* **What**: CNNs trained on scene-specific data to classify foreground pixels.
* **Use Case**: Semantic segmentation or robust detection in high-noise scenes.
* **Complement to GMM**:
 * GMM can act as a **fast filter**, with DL providing **object class refinement**.
 * Useful for labeling objects after motion detection (e.g., plane, cloud, bird).

### Other Useful Techniques

| Technique                                | Description                                         | Benefit                                                        |
| ---------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| **Shadow Removal (Chromaticity Models)** | Analyze color ratios or HSV components              | Reduce false positives in GMM                                  |
| **Edge Consistency Check**               | Use edge maps (Sobel, Canny)                        | Confirm object boundaries                                      |
| **Temporal Consistency Filter**          | Use a **temporal buffer** to track object stability | Filter out transient noise or blinking                         |
| **Noise Map Calibration**                | Estimate per-pixel variance floor from night scenes | Prevent overreacting to photon shot noise or sensor hot pixels |
| **Temporal Median Filter**               | Keeps a median over past N frames                   | Strong static background model alternative                     |


## Designing a Hybrid Pipeline

Depending on the application, combining GMM with these techniques can significantly improve performance:

* **Fast Motion Detection**: GMM + 3-frame difference
* **Ghost Suppression**: GMM + single-Gaussian + background mask
* **Sky/Star Tracking**: GMM + DoG + contrast enhancement
* **Smoke/Contrails**: GMM + Gabor filters
* **Illumination Invariance**: GMM + gradient-based difference or optical flow


