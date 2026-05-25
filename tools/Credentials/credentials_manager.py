import json
import hashlib
import os
import csv
from pathlib import Path
import secrets


# ---------- Config ----------
def get_config_dir():
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "sky360"
    else:
        return Path.home() / ".config" / "sky360"


CONFIG_FILE = get_config_dir() / "sky360_list_of_trusted_users.json"
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------- Storage ----------
def load_users():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_users(users):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)


# ---------- Helpers ----------
def stable_u64(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def generate_credentials():
    return secrets.token_urlsafe(32)


def compute_ids(user):
    identity = f"{user['name']}|{user['email']}"
    user["DataID"] = stable_u64(identity)
    # DataSignature is NOT stored anymore


def find_user(users, key, value):
    return next((u for u in users if str(u.get(key)) == str(value)), None)


# ---------- Core ----------
def add_user():
    users = load_users()

    name = input("Name: ").strip()
    email = input("Email: ").strip()

    credentials = generate_credentials()

    user = {
        "name": name,
        "email": email,
        "credentials": credentials
    }

    compute_ids(user)
    users.append(user)
    save_users(users)

    print("\n✅ User added")
    print(f"Credentials: {credentials}")


def update_user():
    users = load_users()

    key = input("Search by (name/email): ").strip()
    value = input("Value: ").strip()

    user = find_user(users, key, value)

    if not user:
        print("❌ User not found")
        return

    new_name = input(f"New name [{user['name']}]: ").strip() or user["name"]
    new_email = input(f"New email [{user['email']}]: ").strip() or user["email"]

    regenerate = input("Regenerate credentials? (y/n): ").lower() == "y"

    user["name"] = new_name
    user["email"] = new_email

    if regenerate:
        user["credentials"] = generate_credentials()

    compute_ids(user)
    save_users(users)

    print("✅ User updated")


def search_user():
    users = load_users()

    print("Search by: name, email, DataID")
    key = input("Field: ").strip()
    value = input("Value: ").strip()

    user = find_user(users, key, value)

    if user:
        print("\n✅ Found:")
        print(json.dumps(user, indent=4))
    else:
        print("❌ No match found")


# ✅ NEW FUNCTION
def validate_data():
    users = load_users()

    print("\n--- Validate Incoming Data ---")

    data_id = int(input("DataID: ").strip())
    data_sig = int(input("DataSignature: ").strip())

    epoch_sec = input("epoch time sec: ").strip()
    epoch_nsec = input("epoch time nsec: ").strip()
    mono_sec = input("mono time sec: ").strip()
    mono_nsec = input("mono time nsec: ").strip()

    epoch_str = f"{epoch_sec}|{epoch_nsec}"
    mono_str = f"{mono_sec}|{mono_nsec}"

    # Step 1: find user
    user = find_user(users, "DataID", data_id)

    if not user:
        print("❌ No user found for this DataID")
        return

    # Step 2: recompute signature
    signature_input = f"{user['credentials']}|{epoch_str}|{mono_str}"
    expected_sig = stable_u64(signature_input)

    # Step 3: compare
    if expected_sig == data_sig:
        print("✅ VALID DATA")
    else:
        print("❌ INVALID SIGNATURE")

        print("\nExpected:", expected_sig)
        print("Received:", data_sig)


def export_csv():
    users = load_users()

    if not users:
        print("No users to export")
        return

    csv_file = CONFIG_FILE.with_suffix(".csv")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=users[0].keys())
        writer.writeheader()
        writer.writerows(users)

    print(f"✅ Exported to: {csv_file}")


# ---------- CLI ----------
def main():
    while True:
        print("\n--- SKY360 Credential Manager ---")
        print("1. Add user")
        print("2. Update user")
        print("3. Search user")
        print("4. Validate data")
        print("5. Export CSV")
        print("6. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            add_user()
        elif choice == "2":
            update_user()
        elif choice == "3":
            search_user()
        elif choice == "4":
            validate_data()
        elif choice == "5":
            export_csv()
        elif choice == "6":
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()