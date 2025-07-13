import multiprocessing
import time
import os
import psutil
import numpy as np
from threading import Thread

# -------- CPU TEST --------
def burn_cpu(core_id):
    os.sched_setaffinity(0, {core_id})
    while True:
        _ = 123456789 ** 0.5  # Arbitrary calculation

# -------- MEMORY TEST --------
def burn_memory(size_gb):
    print(f"Allocating {size_gb} GB in memory...")
    big_array = np.zeros((size_gb * 1024 * 1024 * 1024 // 8,), dtype=np.float64)
    time.sleep(60)  # Hold for 60s

# -------- SWAP TEST --------
def burn_memory_until_swap(threshold_gb):
    print("Allocating memory until swap is triggered...")
    chunks = []
    try:
        while True:
            chunks.append(bytearray(512 * 1024 * 1024))  # 512MB chunk
            mem = psutil.virtual_memory()
            print(f"RAM used: {mem.percent}%, Swap: {psutil.swap_memory().percent}%")
            if mem.percent > threshold_gb * 100:
                break
    except MemoryError:
        print("MemoryError triggered (RAM full)")

    time.sleep(30)

# -------- GPU TEST --------
def stress_gpu():
    try:
        import pyopencl as cl
        import pyopencl.array
        import numpy as np

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        a_np = np.random.rand(5000000).astype(np.float32)
        b_np = np.random.rand(5000000).astype(np.float32)

        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        dest_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

        prg = cl.Program(ctx, """
        __kernel void sum(__global const float *a,
                          __global const float *b,
                          __global float *c)
        {
          int gid = get_global_id(0);
          c[gid] = a[gid] + b[gid];
        }
        """).build()

        print("Starting GPU stress test...")
        while True:
            prg.sum(queue, a_np.shape, None, a_g, b_g, dest_g)
            queue.finish()
    except Exception as e:
        print("GPU stress failed:", e)

# -------- NPU TEST (STUB) --------
def test_npu_stub():
    print("Simulating NPU load (stub) — requires SDK like RKNN...")
    time.sleep(10)

# -------- RGA TEST (STUB) --------
def test_rga_stub():
    import cv2
    print("Simulating RGA load using OpenCV resize...")
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    while True:
        resized = cv2.resize(img, (640, 480))
        resized = cv2.resize(resized, (1920, 1080))

# -------- RUNNER --------
if __name__ == "__main__":
    num_cores = os.cpu_count()

    print("Starting CPU load on all cores...")
    cpu_processes = []
    for core_id in range(num_cores):
        p = multiprocessing.Process(target=burn_cpu, args=(core_id,))
        p.start()
        cpu_processes.append(p)

    # Memory test (e.g., 2GB)
    mem_thread = Thread(target=burn_memory, args=(2,))
    mem_thread.start()

    # Swap stress (threshold at ~90% RAM used)
    swap_thread = Thread(target=burn_memory_until_swap, args=(0.90,))
    swap_thread.start()

    # RGA test
    rga_thread = Thread(target=test_rga_stub)
    rga_thread.start()

    # GPU test
    gpu_thread = Thread(target=stress_gpu)
    gpu_thread.start()

    # NPU test (only works if SDK installed)
    npu_thread = Thread(target=test_npu_stub)
    npu_thread.start()

    # Wait for threads
    mem_thread.join()
    swap_thread.join()
