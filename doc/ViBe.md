**ViBe** is a background subtraction algorithm designed for motion detection in video sequences. It introduces several innovative mechanisms to improve efficiency and accuracy:

- **Sample-Based Background Model:** Instead of maintaining a single background model, ViBe stores multiple past pixel values at each location or in the neighborhood.
- **Randomized Model Update:** Unlike traditional methods that replace the oldest values first, ViBe randomly selects which background values to update, enhancing adaptability.
- **Spatial Propagation:** When a pixel is classified as background, its value is propagated to neighboring pixels, improving consistency.
- **Computational Efficiency:** ViBe is optimized for speed, using simple operations like subtractions, comparisons, and memory manipulation.

### **ViBe: Summary**

**Goal**:
ViBe (Visual Background extractor) is an efficient and robust **background subtraction algorithm** designed to detect **moving objects in video sequences**. It is intended to work universally across various scenes with minimal parameter tuning.

### **Key Ideas**:

1. **Background Model Per Pixel**:
   * Each pixel maintains a **sample set** of intensity values from previous frames, rather than a single estimated value.
   * This set acts as a **non-parametric model** capturing the variability of the background at that pixel.


2. **Foreground Detection**:
   * A pixel is classified as foreground if **its current value differs significantly** from most values in its sample set.
   * The decision is based on whether the number of close (within a threshold) values in the sample set exceeds a minimum match count.


3. **Model Update**:
   * The model is updated **conservatively** and **randomly**:
     * With a certain probability, a background pixel’s value is added to its sample set.
     * Neighboring pixels’ models are also randomly updated, promoting **spatial coherence**.


4. **Advantages**:
   * **Fast and lightweight** – suitable for real-time applications.
   * Robust against **noise**, **illumination changes**, and **background motion**.
   * Works well without scene-specific tuning, making it **universal**.


5. **No Explicit Learning Phase**:
   * The model starts working from the very first frame with no need for a long training period.


ViBe uses a **buffer-based approach** rather than a simple running average. It maintains a **set of past pixel values** for each location, stored in a buffer. When determining whether a pixel belongs to the background, ViBe compares the current pixel value against a randomly selected subset of stored values.

Unlike running average methods, which continuously update a mean value, ViBe **randomly replaces stored pixel values** to ensure adaptability while avoiding abrupt changes. This randomness helps maintain a diverse background model, making it more robust to sudden scene variations.

The **buffer-based approach of ViBe** could be efficient on the **Orange Pi 5+**, but it depends on the specific constraints of the application. ViBe is lightweight in terms of computation because it relies on simple pixel comparisons and random updates rather than complex probability distributions. However, its **memory footprint** can be higher due to the need to store multiple past pixel values per location.

### **Comparison with Zivkovic’s GMM**
- **ViBe (Buffer-Based)**
  - **Pros:** Fast pixel-wise comparisons, adaptable to sudden changes, low computational overhead.
  - **Cons:** Requires more memory per pixel, less statistical modeling of background variations.

- **Zivkovic’s Adaptive GMM**
  - **Pros:** Dynamically adjusts the number of Gaussian components per pixel, better at handling gradual illumination changes.
  - **Cons:** Higher computational cost due to probability updates, may struggle with sudden scene changes.

### **Efficiency on Orange Pi 5+**
- **ViBe** could be more efficient for **real-time applications** on Orange Pi 5+ due to its simplicity and lower CPU usage.
- **GMM** might be preferable if **background variations are complex** and require statistical modeling, but it could be **more demanding** on the Orange Pi’s processing power.

### **ViBe vs Zivkovic’s GMM (MOG2)**

| Feature / Property                  | **ViBe**                                        | **Zivkovic’s GMM (MOG2)**                        |
| ----------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| **Background Model Type**           | Sample-based model (pixel history buffer)       | Parametric model (Gaussian mixture)              |
| **Update Mechanism**                | Randomized replacement of samples per pixel     | Adaptive learning rate with online updates       |
| **Storage per Pixel**               | High (buffer of N previous values)              | Moderate (K Gaussian parameters)                 |
| **Motion Adaptation**               | Sensitive to motion; fast to adapt              | Slower to adapt, better for slow dynamic scenes  |
| **Illumination Robustness**         | Medium – can struggle with gradual light change | Good – statistical adaptation to gradual changes |
| **Dynamic Backgrounds (e.g., sky)** | Weak – high false positives in cloud dynamics   | Stronger – multiple Gaussians handle variations  |
| **Shadow Detection**                | No built-in support                             | Yes – shadow labeling via color model            |
| **Foreground Blobs Stability**      | Noisy – needs post-processing                   | Smoother – more stable segmentation              |
| **Noise Sensitivity**               | High – needs filtering (e.g., QCC)              | Lower – models noise statistically               |
| **Initialization Time**             | Very fast                                       | Slower – needs some training frames              |
| **Compute Load**                    | Lower per frame                                 | Higher (due to float ops and matching)           |
| **OpenCV Support**                  | Not native (external or custom implementation)  | Built-in as `cv::BackgroundSubtractorMOG2`       |
| **Best For**                        | Static cameras, fast moving objects             | Complex, dynamic scenes like water, sky, snow    |
| **Weaknesses**                      | Fails under dynamic backgrounds or slow changes | May blur fast/small objects; tuning is critical  |

---

### Suitability for Sky Observer

| Scenario                                | **ViBe**                          | **Zivkovic GMM (MOG2)**                 |
| --------------------------------------- | --------------------------------- | --------------------------------------- |
| **Changing Sky Conditions**             | ❌ Struggles with transitions      | ✅ Adapts with Gaussian mixtures         |
| **Day/Night Switch**                    | ❌ Needs external model switching  | ✅ Can slowly adapt (or be switched too) |
| **Snow/Rain Events**                    | ❌ High false detections           | ✅ Handles with multi-modal background   |
| **Fast Drones in Clear Sky**            | ✅ Quick detection possible        | ✅ Good if configured with low variance  |
| **Small Object Tracking**               | ⚠️ Needs noise filtering          | ✅ Better consistency over time          |
| **Edge-case Illumination (Sun, glare)** | ❌ Requires color normalization    | ✅ More stable with shadow model         |
| **Long-Term Operation (24/7)**          | ⚠️ Needs periodic resets/cleaning | ✅ More robust, but may still drift      |

---

### Recommendation for Sky Observer:

| Use Case                                      | Recommendation                                     |
| --------------------------------------------- | -------------------------------------------------- |
| **Robust 24/7 operation**                     | ✅ **MOG2 (Zivkovic)** with tuning                  |
| **Detecting small, fast anomalies**           | ⚠️ **ViBe** + post-processing (e.g., QCC, filters) |
| **Dynamic scenes (clouds, snow, light)**      | ✅ GMM wins with its adaptive background model      |
| **Low compute footprint (e.g., edge device)** | ⚠️ ViBe is lighter but needs cleaning              |

---

### Hybrid Option

* Combine **Zivkovic GMM** for stable background
* Use **gradient-based or motion filters** for detecting **drones**
* Consider your **GMM of GMMs** switcher idea for **night/day/cloud** profiles
* Add **QCC or morphological cleanup** for foreground

For **QHY183C and a 180° fisheye**, background subtraction method needs to handle **dynamic lighting, weather changes, and small moving objects** efficiently.

| Feature | **ViBe (Buffer-Based)** | **Zivkovic’s Adaptive GMM** |
|---------|------------------------|----------------------------|
| **Background Model** | Stores multiple past pixel values in a buffer | Uses a mixture of Gaussians to model pixel intensity variations |
| **Adaptability** | Fast adaptation to sudden changes (e.g., clouds, rain) | Gradual adaptation, better for slow illumination changes |
| **Memory Usage** | Higher due to stored pixel samples | Lower, but computationally heavier due to probability updates |
| **Computational Cost** | Low (simple pixel comparisons) | Higher (Gaussian updates and probability calculations) |
| **Handling Small Objects** | Can detect small objects well if tuned properly | More robust for small objects with gradual motion |
| **Weather Resilience** | Handles abrupt changes well but may struggle with persistent noise | More stable under gradual environmental shifts |
| **Real-Time Performance on Orange Pi 5+** | Efficient due to lightweight operations | May require optimization for real-time execution |

