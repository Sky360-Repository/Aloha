import hashlib
import json
import os
import time
import secrets
from pathlib import Path

class Credentials:
    def __init__(self):
        self.name = ""
        self.email = ""
        self.credentials = ""
        self.valid = False
        self.error = None

        self._load_local_user_credentials()



    def apply_to_header(self, header):
        """
        Writes timestamps + DataID + DataSignature.
        Never raises if credentials are missing.
        """

        # ---- Create timestamps ----

        # epoch time (wall clock)
        now_ns = time.time_ns()
        epoch_time = header.epoch_time.__class__()  # create same type
        epoch_time.sec = now_ns // 1_000_000_000
        epoch_time.nsec = now_ns % 1_000_000_000

        # monotonic time
        mono_ns = time.monotonic_ns()
        mono_time = header.mono_time.__class__()
        mono_time.sec = mono_ns // 1_000_000_000
        mono_time.nsec = mono_ns % 1_000_000_000

        # ---- Assign to header ----
        header.epoch_time.CopyFrom(epoch_time)
        header.mono_time.CopyFrom(mono_time)

        # ---- Convert to deterministic strings ----
        epoch_str = f"{epoch_time.sec}|{epoch_time.nsec}"
        mono_str = f"{mono_time.sec}|{mono_time.nsec}"

        # ---- Generate hashes ----
        if self.valid:
            identity = f"{self.name}|{self.email}"
            signature = f"{self.credentials}|{epoch_str}|{mono_str}"

            header.DataID = self._stable_u64(identity)
            header.DataSignature = self._stable_u64(signature)
        else:
            # fallback
            header.DataID = self._random_u64()
            header.DataSignature = self._random_u64()

        return header

    def _load_local_user_credentials(self):
        try:
            cfg_file = self._config_dir() / "trusted_users.json"

            if not cfg_file.exists():
                self.valid = False
                self.error = f"Credentials file not found: {cfg_file}"
                return

            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.valid = False
                self.error = "Invalid trusted_users.json format: expected object"
                return

            self.name = (data.get("name") or "").strip()
            self.email = (data.get("email") or "").strip()
            self.credentials = (data.get("credentials") or "").strip()

            if not self.name or not self.email or not self.credentials:
                self.valid = False
                self.error = (
                    "Missing fields in trusted_users.json. "
                    f"Found: name={bool(self.name)}, email={bool(self.email)}, credentials={bool(self.credentials)}"
                )
                return

            self.valid = True
            self.error = None

        except Exception as e:
            self.valid = False
            self.error = f"Unexpected error reading credentials: {e}"

    @staticmethod
    def _stable_u64(text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _random_u64() -> int:
        return secrets.randbits(64)

    @staticmethod
    def _config_dir() -> Path:
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", str(Path.home())))
        else:
            base = Path.home() / ".config"
        return base / "sky360"
