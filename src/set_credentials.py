import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import json
import os

# ---- Config path (cross-platform safe) ----
def get_config_path():
    if os.name == "nt":
        base = Path(os.environ["APPDATA"])
    else:
        base = Path.home() / ".config"
    return base / "sky360"

CONFIG_DIR = get_config_path()
CONFIG_FILE = CONFIG_DIR / "trusted_users.json"


# ---- Ensure directory exists ----
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ---- Load existing users (if any) ----
def load_users():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


# ---- Save users ----
def save_user(user):
    with open(CONFIG_FILE, "w") as f:
        json.dump(user, f, indent=4)

# ---- GUI logic ----
def save_entry():
    name = entry_name.get().strip()
    email = entry_email.get().strip()
    credentials = entry_credentials.get().strip()

    if not name or not email or not credentials:
        messagebox.showerror("Error", "All fields are required!")
        return

    user = {
        "name": name,
        "email": email,
        "credentials": credentials
    }

    save_user(user)

    messagebox.showinfo("Success", "User saved (overwritten)!")

    # Clear fields
    entry_name.delete(0, tk.END)
    entry_email.delete(0, tk.END)
    entry_credentials.delete(0, tk.END)


# ---- Build GUI ----
root = tk.Tk()
root.iconbitmap(r"..\doc\img\sky360_logo.ico")
root.title("Sky360 - Add Trusted User")
root.geometry("350x250")

tk.Label(root, text="Name").pack(pady=5)
entry_name = tk.Entry(root, width=40)
entry_name.pack()

tk.Label(root, text="Email").pack(pady=5)
entry_email = tk.Entry(root, width=40)
entry_email.pack()

tk.Label(root, text="Credentials (public key / token)").pack(pady=5)
entry_credentials = tk.Entry(root, width=40)
entry_credentials.pack()

tk.Button(root, text="Save User", command=save_entry).pack(pady=20)

root.mainloop()