#!/usr/bin/env python3
"""
Upload models to awAyham/comfyui-models HuggingFace repo.

Usage:
  python upload_to_hf.py --bundle qwen-image-2.1-edit
  python upload_to_hf.py --file vae/qwen_image_2.1_vae_bf16.safetensors
  python upload_to_hf.py --all

Requires: HF_TOKEN environment variable or huggingface-cli login.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from huggingface_hub import HfApi, login

HF_REPO = "awAyham/comfyui-models"
MANIFEST = Path(__file__).parent / "model_manifest.yaml"
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/mnt/user-data/ComfyUI/models"))

def load_manifest():
    with open(MANIFEST) as f:
        return yaml.safe_load(f)

def upload_file(api, local_path, repo_path):
    """Upload a single file to the HF repo."""
    if not local_path.exists():
        print(f"  [SKIP] Not found: {local_path}")
        return False
    
    size_gb = local_path.stat().st_size / 1e9
    print(f"  [UPLOAD] {repo_path} ({size_gb:.2f} GB)...")
    
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=HF_REPO,
            repo_type="model",
        )
        print(f"  [OK] {repo_path}")
        return True
    except Exception as e:
        print(f"  [ERROR] {repo_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload models to HF repo")
    parser.add_argument("--bundle", help="Upload all files in a bundle")
    parser.add_argument("--file", help="Upload a single file by dest path")
    parser.add_argument("--all", action="store_true", help="Upload all files in manifest")
    args = parser.parse_args()
    
    # Check token
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)
    else:
        print("[WARN] No HF_TOKEN set — using cached login")
    
    api = HfApi()
    
    # Ensure repo exists
    try:
        api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
        print(f"[REPO] {HF_REPO} ready")
    except Exception as e:
        print(f"[REPO] Error: {e}")
        return
    
    manifest = load_manifest()
    files_by_dest = {f["dest"]: f for f in manifest["files"]}
    
    # Determine which files to upload
    to_upload = []
    if args.bundle:
        if args.bundle not in manifest["bundles"]:
            print(f"[ERROR] Bundle not found: {args.bundle}")
            print(f"Available: {list(manifest['bundles'].keys())}")
            return
        to_upload = manifest["bundles"][args.bundle]
    elif args.file:
        to_upload = [args.file]
    elif args.all:
        to_upload = [f["dest"] for f in manifest["files"]]
    else:
        print("[ERROR] Specify --bundle, --file, or --all")
        return
    
    print(f"\n[UPLOAD] {len(to_upload)} files to {HF_REPO}\n")
    
    success = 0
    failed = 0
    skipped = 0
    
    for dest in to_upload:
        if dest not in files_by_dest:
            print(f"  [SKIP] Not in manifest: {dest}")
            skipped += 1
            continue
        
        local_path = MODELS_ROOT / dest
        if upload_file(api, local_path, dest):
            success += 1
        elif not local_path.exists():
            skipped += 1
        else:
            failed += 1
    
    print(f"\n[DONE] {success} uploaded · {failed} failed · {skipped} skipped")

if __name__ == "__main__":
    main()
