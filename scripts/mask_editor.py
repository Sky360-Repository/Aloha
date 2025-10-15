#!/usr/bin/env python3
# coding: utf-8

# \copyright    Sky360.org
# \co-author    ChatGPT4
#
# \brief        Editor for the mask.png file
#
# ************************************************************************

"""
- Loads current frame from shared memory ring buffer at startup.
- Loads mask.png if present (3200x3200); otherwise creates white mask (255 = background).
- Paint (LMB) = set mask to 0 (foreground). Erase (RMB) = set mask to 255.
- Shift+click connects last click to new click with a straight line.
- Ctrl+LMB performs flood-fill on mask (toggleable).
- Undo/Redo stack.
- Zoom with mouse wheel. Pan with middle mouse button.
- Save writes mask.png containing 0 for foreground and 255 for background.
"""

import os
import sys
from collections import deque
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from multiprocessing import shared_memory, resource_tracker

# ----------------------- Configuration -----------------------
RING_BUFFER_NAME = "cam_ring_buffer"
METADATA_BUFFER_NAME = "metadata_ring_buffer"

FRAME_WIDTH = 3200
FRAME_HEIGHT = 3200
FRAME_COUNT = 200

DISPLAY_SIZE = 800
PEN_RADIUS_DEFAULT = 50
TRANSPARENCY_DEFAULT = 40
UNDO_STACK_SIZE = 20

MASK_FILENAME = "mask.png"

# ------------------- Shared memory helpers -------------------
def attach_shared_memory():
    shm = shared_memory.SharedMemory(name=RING_BUFFER_NAME)
    resource_tracker.unregister(shm._name, 'shared_memory')
    frame_buffer = np.ndarray((FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint16, buffer=shm.buf)

    shm2 = shared_memory.SharedMemory(name=METADATA_BUFFER_NAME)
    resource_tracker.unregister(shm2._name, 'shared_memory')
    metadata_buffer = np.ndarray((FRAME_COUNT, 3), dtype=np.float64, buffer=shm2.buf)

    return shm, shm2, frame_buffer, metadata_buffer

def get_newest_index(metadata_buffer):
    return int(np.argmax(metadata_buffer[:, 0]))

# ------------------- Mask drawing helpers -------------------
def mask_draw_circle(mask, cx, cy, radius, value):
    cv2.circle(mask, (cx, cy), radius, int(value), thickness=-1)

def mask_draw_line(mask, x0, y0, x1, y1, radius, value):
    dx, dy = x1 - x0, y1 - y0
    dist = int(np.hypot(dx, dy))
    if dist == 0:
        mask_draw_circle(mask, x0, y0, radius, value)
        return
    steps = max(1, dist // max(1, radius // 2))
    for t in range(steps + 1):
        alpha = t / steps
        xi = int(round(x0 + dx * alpha))
        yi = int(round(y0 + dy * alpha))
        mask_draw_circle(mask, xi, yi, radius, value)

def mask_flood_fill(mask, x, y, target_value, replacement_value):
    """
    Flood fill contiguous region in mask in-place.
    Uses OpenCV floodFill on the actual mask so the operation is atomic and undoable.
    """
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return
    if mask[y, x] != target_value or target_value == replacement_value:
        return
    # mask must be uint8
    if mask.dtype != np.uint8:
        raise ValueError("mask must be uint8")
    # temporary mask for floodFill requires 2 pixels larger
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    # flags: 4-connectivity + fill value in the high byte
    flags = 4 | (255 << 8)
    cv2.floodFill(mask, ff_mask, (x, y), int(replacement_value), loDiff=0, upDiff=0, flags=flags)

# ------------------- Main Tkinter Editor -------------------
class MaskEditorApp:
    def __init__(self, root, bg_image_rgb, mask_full_res, display_size=DISPLAY_SIZE):
        self.root = root
        self.root.title("Mask Editor")
        self.mask = mask_full_res.copy()
        self.bg_display = bg_image_rgb
        self.full_h, self.full_w = self.mask.shape

        # view transform
        self.view_size = display_size
        self.base_scale = self.view_size / max(self.full_w, self.full_h)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # pen & UI
        self.pen_radius = tk.IntVar(value=PEN_RADIUS_DEFAULT)
        self.transparency = tk.IntVar(value=TRANSPARENCY_DEFAULT)

        # undo/redo
        self.undo_stack = deque(maxlen=UNDO_STACK_SIZE)
        self.redo_stack = deque(maxlen=UNDO_STACK_SIZE)
        self.push_undo()  # initial snapshot

        # drawing state
        self.is_drawing = False
        self.drawing_mode = 'paint'  # paint => 0; erase => 255
        self.last_mask_pos = None
        self.last_canvas_click = None

        # cached scaled mask for display speed
        self.mask_disp = None
        self.mask_disp_size = (0, 0)
        self.recompute_mask_disp()

        # refresh scheduling
        self._refresh_scheduled = False

        # build UI
        self.create_widgets()
        self.bind_events()
        self.refresh_display()

    # ---------------- Undo/Redo ----------------
    def push_undo(self):
        # append current mask snapshot and clear redo (committed operation)
        self.undo_stack.append(self.mask.copy())
        self.redo_stack.clear()

    def do_undo(self):
        if len(self.undo_stack) <= 1:
            return
        self.redo_stack.append(self.undo_stack.pop())
        self.mask[:] = self.undo_stack[-1]
        self.recompute_mask_disp()
        self.schedule_refresh()

    def do_redo(self):
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        self.undo_stack.append(state.copy())
        self.mask[:] = state
        self.recompute_mask_disp()
        self.schedule_refresh()

    # ---------------- UI ----------------
    def create_widgets(self):
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.pack(side='left', fill='both', expand=True)
        self.right_frame = ttk.Frame(self.root, width=260)
        self.right_frame.pack(side='right', fill='y')

        self.canvas = tk.Canvas(self.left_frame, width=self.view_size, height=self.view_size,
                                bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.pil_bg = Image.fromarray(self.bg_display)
        self.tk_bg = ImageTk.PhotoImage(self.pil_bg)
        self.bg_image_id = self.canvas.create_image(0, 0, anchor='nw', image=self.tk_bg)
        self.tk_display = None

        # Top row: Undo / Redo (side by side)
        ur = ttk.Frame(self.right_frame)
        ur.pack(pady=(8,4), padx=6)
        ttk.Button(ur, text="Undo", width=10, command=self.do_undo).pack(side='left', padx=2)
        ttk.Button(ur, text="Redo", width=10, command=self.do_redo).pack(side='left', padx=2)

        # Second row: View All / Zoom Max
        vz = ttk.Frame(self.right_frame)
        vz.pack(pady=(4,8), padx=6)
        ttk.Button(vz, text="View All", width=10, command=self.view_all).pack(side='left', padx=2)
        ttk.Button(vz, text="Zoom Max", width=10, command=self.zoom_max).pack(side='left', padx=2)

        ttk.Separator(self.right_frame).pack(fill='x', pady=6, padx=6)

        ttk.Label(self.right_frame, text="Pen Size").pack(pady=(6,2), anchor='w', padx=8)
        pen_scale = ttk.Scale(self.right_frame, from_=1, to=400, orient='horizontal',
                              variable=self.pen_radius, command=lambda e: self.schedule_refresh())
        pen_scale.pack(fill='x', padx=8)

        ttk.Label(self.right_frame, text="Overlay Transparency (%)").pack(pady=(8,2), anchor='w', padx=8)
        ttk.Scale(self.right_frame, from_=0, to=100, orient='horizontal',
                  variable=self.transparency, command=lambda e: self.schedule_refresh()).pack(fill='x', padx=8)

        ttk.Separator(self.right_frame).pack(fill='x', pady=6, padx=6)

        # Reload / Clear side-by-side
        rc = ttk.Frame(self.right_frame)
        rc.pack(pady=(6,4), padx=6, fill='x')
        ttk.Button(rc, text="Reload", command=self.reload_mask).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(rc, text="Clear", command=self.clear_mask).pack(side='left', expand=True, fill='x', padx=2)

        ttk.Separator(self.right_frame).pack(fill='x', pady=6, padx=6)

        ttk.Label(self.right_frame, text="Hints:").pack(anchor='w', padx=8)
        hints = [
            "LMB = paint (set 0)",
            "RMB = erase (set 255)",
            "Shift+click = straight line",
            "Ctrl+LMB = flood-fill region",
            "Mouse wheel = zoom",
            "Middle-button drag = pan",
            "Close window to quit"
        ]
        for h in hints:
            ttk.Label(self.right_frame, text=h).pack(anchor='w', padx=12)

        # spacer and save at bottom
        ttk.Label(self.right_frame, text="").pack(expand=True, fill='y')
        ttk.Button(self.right_frame, text="Save", command=self.save_mask).pack(side='bottom', fill='x', padx=8, pady=8)

    def bind_events(self):
        # drawing
        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        self.canvas.bind("<Button-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_up)

        # pan
        self.canvas.bind("<Button-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)

        # wheel (Windows/macOS) and linux buttons (4/5)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        # keyboard
        self.root.bind("<Control-z>", lambda e: self.do_undo())
        self.root.bind("<Control-y>", lambda e: self.do_redo())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # cursor circle
        self.canvas.bind("<Motion>", self.on_mouse_move)

    # ---------------- Coordinate transforms ----------------
    def view_to_mask(self, vx, vy):
        eff = self.base_scale * self.zoom
        mx = int(round((vx - self.pan_x) / eff))
        my = int(round((vy - self.pan_y) / eff))
        mx = max(0, min(self.full_w - 1, mx))
        my = max(0, min(self.full_h - 1, my))
        return mx, my

    def mask_to_view(self, mx, my):
        eff = self.base_scale * self.zoom
        vx = int(round(mx * eff + self.pan_x))
        vy = int(round(my * eff + self.pan_y))
        return vx, vy

    # ---------------- Mouse handlers ----------------
    def on_left_down(self, event):
        ctrl = (event.state & 0x0004) != 0 or (event.state & 0x0008) != 0
        shift = (event.state & 0x0001) != 0
        if ctrl:
            mx, my = self.view_to_mask(event.x, event.y)
            target = int(self.mask[my, mx])
            replacement = 255 if self.drawing_mode == 'erase' else 0
            self.push_undo()
            mask_flood_fill(self.mask, mx, my, target, replacement)
            self.recompute_mask_disp()
            self.schedule_refresh()
            return
        if shift and self.last_canvas_click is not None:
            x0, y0 = self.last_canvas_click
            x1, y1 = event.x, event.y
            m0x, m0y = self.view_to_mask(x0, y0)
            m1x, m1y = self.view_to_mask(x1, y1)
            val = 255 if self.drawing_mode == 'erase' else 0
            self.push_undo()
            mask_draw_line(self.mask, m0x, m0y, m1x, m1y, self.pen_radius.get(), val)
            self.recompute_mask_disp()
            self.last_canvas_click = (event.x, event.y)
            self.schedule_refresh()
            return
        # freehand start
        self.is_drawing = True
        self.last_canvas_click = (event.x, event.y)
        mx, my = self.view_to_mask(event.x, event.y)
        val = 255 if self.drawing_mode == 'erase' else 0
        # do not push_undo here; push at mouse up to make stroke atomic
        mask_draw_circle(self.mask, mx, my, self.pen_radius.get(), val)
        self.recompute_mask_disp()
        self.last_mask_pos = (mx, my)
        self.schedule_refresh()

    def on_left_drag(self, event):
        if not self.is_drawing:
            return
        mx, my = self.view_to_mask(event.x, event.y)
        if self.last_mask_pos is None:
            self.last_mask_pos = (mx, my)
        x0, y0 = self.last_mask_pos
        x1, y1 = mx, my
        val = 255 if self.drawing_mode == 'erase' else 0
        mask_draw_line(self.mask, x0, y0, x1, y1, self.pen_radius.get(), val)
        self.last_mask_pos = (mx, my)
        self.last_canvas_click = (event.x, event.y)
        self.recompute_mask_disp()
        self.schedule_refresh()

    def on_left_up(self, event):
        # commit stroke as a single undo snapshot
        if self.is_drawing:
            self.push_undo()
        self.is_drawing = False
        self.last_mask_pos = None

    def on_right_down(self, event):
        # treat right-click as erase shortcut (temporary mode flip)
        prev = self.drawing_mode
        self.drawing_mode = 'erase'
        self.on_left_down(event)
        self.drawing_mode = prev

    def on_right_drag(self, event):
        prev = self.drawing_mode
        self.drawing_mode = 'erase'
        self.on_left_drag(event)
        self.drawing_mode = prev

    def on_right_up(self, event):
        self.on_left_up(event)

    def on_middle_down(self, event):
        self.pan_last = (event.x, event.y)

    def on_middle_drag(self, event):
        if self.pan_last is None:
            return
        dx = event.x - self.pan_last[0]
        dy = event.y - self.pan_last[1]
        self.pan_x += dx
        self.pan_y += dy
        self.pan_last = (event.x, event.y)
        # don't change mask_disp on pan (only view changes)
        self.schedule_refresh()

    def on_middle_up(self, event):
        self.pan_last = None

    def on_mouse_wheel(self, event):
        # handle wheel on different platforms
        if hasattr(event, 'delta') and event.delta:
            delta = event.delta
        else:
            if event.num == 4:
                delta = 120
            elif event.num == 5:
                delta = -120
            else:
                delta = 0
        factor = 1.1 if delta > 0 else 0.9
        mx_view, my_view = event.x, event.y
        before_mx, before_my = self.view_to_mask(mx_view, my_view)
        self.zoom = max(0.05, min(20.0, self.zoom * factor))
        # adjust pan so the mask point under cursor stays fixed
        eff = self.base_scale * self.zoom
        new_vx = before_mx * eff + self.pan_x
        new_vy = before_my * eff + self.pan_y
        self.pan_x += mx_view - new_vx
        self.pan_y += my_view - new_vy
        # recompute mask_disp for new zoom
        self.recompute_mask_disp()
        self.schedule_refresh()

    def on_mouse_move(self, event):
        # draw non-destructive cursor circle
        self.draw_cursor(event)

    # ---------------- Mask & Rendering ----------------
    def recompute_mask_disp(self):
        eff = self.base_scale * self.zoom
        disp_w = int(round(self.full_w * eff))
        disp_h = int(round(self.full_h * eff))
        if disp_w <= 0 or disp_h <= 0:
            self.mask_disp = None
            self.mask_disp_size = (0, 0)
            return
        self.mask_disp = cv2.resize(self.mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        self.mask_disp_size = (disp_w, disp_h)

    def refresh_display(self):
        eff = self.base_scale * self.zoom
        disp_w = int(round(self.full_w * eff))
        disp_h = int(round(self.full_h * eff))

        # Resize background from original PIL (which was display-size based initially).
        # For better zoom quality you may want to keep a full-res background and resample from it.
        bg_resized = self.pil_bg.resize((disp_w, disp_h), resample=Image.BILINEAR)
        canvas_img = Image.new("RGBA", (self.view_size, self.view_size), (0,0,0,255))
        canvas_img.paste(bg_resized, (int(round(self.pan_x)), int(round(self.pan_y))))

        canvas_np = np.array(canvas_img)  # RGBA

        # overlay from mask_disp cache
        overlay = np.zeros((self.view_size, self.view_size, 4), dtype=np.uint8)
        if self.mask_disp is None or self.mask_disp_size != (disp_w, disp_h):
            self.recompute_mask_disp()
        if self.mask_disp is not None:
            x0 = int(round(self.pan_x))
            y0 = int(round(self.pan_y))
            x1 = max(0, x0)
            y1 = max(0, y0)
            x2 = min(self.view_size, x0 + disp_w)
            y2 = min(self.view_size, y0 + disp_h)
            if x1 < x2 and y1 < y2:
                src_x1 = x1 - x0
                src_y1 = y1 - y0
                src_x2 = src_x1 + (x2 - x1)
                src_y2 = src_y1 + (y2 - y1)
                submask = self.mask_disp[src_y1:src_y2, src_x1:src_x2]
                fg = (submask == 0)
                if fg.any():
                    alpha_pct = self.transparency.get()
                    alpha_val = int(round(255 * (alpha_pct / 100.0)))
                    overlay[y1:y2, x1:x2, 0][fg] = 255
                    overlay[y1:y2, x1:x2, 1][fg] = 0
                    overlay[y1:y2, x1:x2, 2][fg] = 255
                    overlay[y1:y2, x1:x2, 3][fg] = alpha_val

        # alpha blend
        alpha_o = (overlay[:, :, 3:4].astype(np.float32) / 255.0)
        if alpha_o.max() > 0:
            canvas_np[:, :, :3] = (overlay[:, :, :3].astype(np.float32) * alpha_o +
                                   canvas_np[:, :, :3].astype(np.float32) * (1 - alpha_o)).astype(np.uint8)

        final = Image.fromarray(canvas_np[:, :, :3])
        self.tk_display = ImageTk.PhotoImage(final)
        self.canvas.itemconfig(self.bg_image_id, image=self.tk_display)

        # draw cursor circle on top (non-destructive)
        self.draw_cursor(None)

    def draw_cursor(self, event):
        # remove previous circle if exists
        if hasattr(self, "_cursor_id") and self._cursor_id is not None:
            try:
                self.canvas.delete(self._cursor_id)
            except Exception:
                pass
            self._cursor_id = None
        # position
        try:
            if event is None:
                mx = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
                my = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
            else:
                mx, my = event.x, event.y
        except Exception:
            return
        if mx < 0 or my < 0 or mx >= self.view_size or my >= self.view_size:
            return
        r = max(1, int(round(self.pen_radius.get() * (self.base_scale * self.zoom))))
        self._cursor_id = self.canvas.create_oval(mx - r, my - r, mx + r, my + r, outline='lime', width=1)

    # ---------------- View controls ----------------
    def view_all(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.recompute_mask_disp()
        self.schedule_refresh()

    def zoom_max(self):
        self.zoom = 4.0
        self.recompute_mask_disp()
        self.schedule_refresh()

    # ---------------- File ops ----------------
    def save_mask(self):
        try:
            cv2.imwrite(MASK_FILENAME, self.mask)
            messagebox.showinfo("Saved", f"Saved {MASK_FILENAME}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def reload_mask(self):
        if not os.path.exists(MASK_FILENAME):
            messagebox.showwarning("Reload", f"{MASK_FILENAME} not found.")
            return
        loaded = cv2.imread(MASK_FILENAME, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            messagebox.showerror("Reload failed", "Failed to read mask.")
            return
        if loaded.shape != self.mask.shape:
            loaded = cv2.resize(loaded, (self.full_w, self.full_h), interpolation=cv2.INTER_NEAREST)
        self.push_undo()
        self.mask[:] = np.where(loaded == 0, 0, 255).astype(np.uint8)
        self.recompute_mask_disp()
        self.schedule_refresh()

    def clear_mask(self):
        if not messagebox.askyesno("Clear", "Clear mask to all background (255)?"):
            return
        self.push_undo()
        self.mask.fill(255)
        self.recompute_mask_disp()
        self.schedule_refresh()

    # ---------------- Refresh scheduling ----------------
    def schedule_refresh(self):
        if not self._refresh_scheduled:
            self._refresh_scheduled = True
            self.root.after_idle(self._do_refresh)

    def _do_refresh(self):
        self._refresh_scheduled = False
        self.refresh_display()

    # ---------------- Shutdown ----------------
    def on_close(self):
        self.root.quit()
        self.root.destroy()

# --------------------------- Main startup -----------------------------------
def main():
    try:
        shm, shm2, frame_buffer, metadata_buffer = attach_shared_memory()
    except Exception as e:
        tk.Tk().withdraw()
        messagebox.showerror("Shared memory error", str(e))
        return

    try:
        idx = get_newest_index(metadata_buffer)
        img16 = frame_buffer[idx]
    except Exception as e:
        try:
            shm.close()
            shm2.close()
        except Exception:
            pass
        tk.Tk().withdraw()
        messagebox.showerror("Frame read error", str(e))
        return

    # try debayer, fallback to gray
    try:
        deb = cv2.cvtColor(img16, cv2.COLOR_BayerRG2RGB)
    except Exception:
        deb = cv2.cvtColor(img16, cv2.COLOR_GRAY2RGB)
    img8 = cv2.convertScaleAbs(deb, alpha=255.0 / 65535.0)
    img_rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)

    # prepare display-sized background (we keep this for simplicity)
    disp = Image.fromarray(img_rgb).resize((DISPLAY_SIZE, DISPLAY_SIZE), resample=Image.BILINEAR)
    display_np = np.array(disp)

    # load mask if exists
    if os.path.exists(MASK_FILENAME):
        loaded = cv2.imread(MASK_FILENAME, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            mask_full = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)
        else:
            if loaded.shape != (FRAME_HEIGHT, FRAME_WIDTH):
                loaded = cv2.resize(loaded, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
            mask_full = np.where(loaded == 0, 0, 255).astype(np.uint8)
    else:
        mask_full = np.full((FRAME_HEIGHT, FRAME_WIDTH), 255, dtype=np.uint8)

    root = tk.Tk()
    app = MaskEditorApp(root, display_np, mask_full, DISPLAY_SIZE)

    try:
        root.mainloop()
    finally:
        try:
            shm.close()
            shm2.close()
        except Exception:
            pass

    print("Mask editor closed.")

if __name__ == "__main__":
    main()

