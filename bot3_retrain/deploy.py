"""Deploy a trained model to the trading bot server via SCP.

Copies the model directory to the server, where ModelWatcher will detect
and hot-swap it into the running Bot3 instance.

Usage:
    python -m bot3_retrain.deploy                          # deploy latest model
    python -m bot3_retrain.deploy models/v_20260428_143000  # deploy specific version
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SERVER_HOST = os.environ.get("DEPLOY_HOST", "YOUR_SERVER_IP")
SERVER_USER = os.environ.get("DEPLOY_USER", "YOUR_SSH_USER")
SERVER_MODEL_DIR = os.environ.get("DEPLOY_MODEL_DIR", "/home/trader/trading-bot/models")
SSH_KEY = os.environ.get(
    "DEPLOY_SSH_KEY",
    str(Path.home() / ".ssh" / "id_ed25519"),
)
SSH_OPTS = ["-i", SSH_KEY, "-o", "StrictHostKeyChecking=no"]


def deploy_model(model_dir: Path) -> bool:
    """SCP a model directory to the trading server.

    Returns True on success, False on failure.
    """
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return False

    required_files = ["classifier.json", "regressor.json", "feature_columns.json", "metadata.json"]
    for fname in required_files:
        if not (model_dir / fname).exists():
            print(f"Error: Missing required file: {model_dir / fname}")
            return False

    version = model_dir.name
    remote_path = f"{SERVER_USER}@{SERVER_HOST}:{SERVER_MODEL_DIR}/{version}"

    print(f"Deploying {version} to {SERVER_HOST}...")
    print(f"  Local:  {model_dir}")
    print(f"  Remote: {SERVER_MODEL_DIR}/{version}")

    # Create remote directory
    mkdir_cmd = [
        "ssh", *SSH_OPTS, f"{SERVER_USER}@{SERVER_HOST}",
        f"mkdir -p {SERVER_MODEL_DIR}/{version}",
    ]
    result = subprocess.run(mkdir_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error creating remote directory: {result.stderr}")
        return False

    # SCP all model files in one call
    local_files = [str(model_dir / f) for f in required_files]
    comparison_file = model_dir / "comparison.json"
    if comparison_file.exists():
        local_files.append(str(comparison_file))

    scp_cmd = [
        "scp", *SSH_OPTS, *local_files,
        f"{SERVER_USER}@{SERVER_HOST}:{SERVER_MODEL_DIR}/{version}/",
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Error copying model files: {result.stderr}")
        # Clean up empty remote dir on failure
        subprocess.run(
            ["ssh", *SSH_OPTS, f"{SERVER_USER}@{SERVER_HOST}",
             f"rmdir {SERVER_MODEL_DIR}/{version} 2>/dev/null"],
            capture_output=True, text=True, timeout=15,
        )
        return False

    # Verify all required files arrived
    verify_cmd = [
        "ssh", *SSH_OPTS, f"{SERVER_USER}@{SERVER_HOST}",
        f"ls {SERVER_MODEL_DIR}/{version}/classifier.json {SERVER_MODEL_DIR}/{version}/regressor.json",
    ]
    result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        print(f"Error: deployed files missing after SCP -- check connectivity")
        return False

    for fname in required_files:
        print(f"  Copied {fname}")

    print(f"\nDeployed successfully. ModelWatcher will detect {version} within 5 minutes.")
    return True


def find_latest_model() -> Path | None:
    """Find the latest model version directory."""
    models_root = Path(__file__).parent.parent / "models"
    versions = sorted(models_root.glob("v_*"))
    return versions[-1] if versions else None


def main():
    parser = argparse.ArgumentParser(description="Deploy ML model to trading server")
    parser.add_argument("model_path", nargs="?", help="Path to model directory (default: latest)")
    args = parser.parse_args()

    if args.model_path:
        model_dir = Path(args.model_path)
    else:
        model_dir = find_latest_model()
        if model_dir is None:
            print("No models found in models/ directory")
            sys.exit(1)

    success = deploy_model(model_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
