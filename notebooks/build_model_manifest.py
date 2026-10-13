#!/usr/bin/env python3
import os, sys, json, yaml
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

MODELS_DIR = Path("/Volumes/App/ComfyUI-models")
OUT = Path(__file__).parent / "model_manifest.yaml"
REPORT = Path(__file__).parent / "manifest_build_report.json"
MIN_SIZE = 100 * 1024 * 1024
EXTS = {".safetensors", ".gguf", ".pt", ".ckpt", ".bin"}

api = HfApi()
hfs = HfFileSystem()
revision_cache = {}

def get_revision(repo_id):
    if repo_id not in revision_cache:
        info = api.repo_info(repo_id, repo_type="model")
        revision_cache[repo_id] = info.sha
    return revision_cache[repo_id]

CANDIDATES = {
    # SD 1.5
    "checkpoints/v1-5-pruned-emaonly.safetensors": [
        ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
    ],
    "vae/vae-ft-mse-840000-ema-pruned.safetensors": [
        ("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors"),
    ],
    # Qwen Image Edit 2511
    "diffusion_models/qwen-image-edit-2511-Q8_0.gguf": [
        ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q8_0.gguf"),
    ],
    "diffusion_models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors": [
        ("lightx2v/Qwen-Image-Edit-2511-Lightning",
         "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors"),
    ],
    "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors": [
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ],
    "vae/qwen_image_vae.safetensors": [
        ("Comfy-Org/Qwen-Image_ComfyUI",
         "split_files/vae/qwen_image_vae.safetensors"),
    ],
    "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors": [
        ("lightx2v/Qwen-Image-Edit-2511-Lightning",
         "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"),
    ],
    "loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors": [
        ("Osrivers/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
         "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"),
    ],
    # FLUX.2 klein
    "diffusion_models/flux-2-klein-4b-Q8_0.gguf": [
        ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q8_0.gguf"),
    ],
    "text_encoders/qwen_3_4b_fp4_flux2.safetensors": [
        ("ModelTrainingSpace/qwen_3_4b_fp4_flux2", "qwen_3_4b_fp4_flux2.safetensors"),
        ("DanielMarius/qwen_3_4b_fp4_flux2.safetensors", "qwen_3_4b_fp4_flux2.safetensors"),
    ],
    "vae/flux2-vae.safetensors": [
        ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
    ],
    # TRELLIS.2 + lotus depth
    "clip_vision/dino_v3_L_naf_fp32.safetensors": [
        ("Comfy-Org/TRELLIS.2", "clip_vision/dino_v3_vit_l.safetensors"),
    ],
    "vae/trellis_2_shape_vae_bf16.safetensors": [
        ("Comfy-Org/TRELLIS.2", "vae/trellis_2_shape_vae_bf16.safetensors"),
    ],
    "vae/trellis_2_texture_vae_bf16.safetensors": [
        ("Comfy-Org/TRELLIS.2", "vae/trellis_2_texture_vae_bf16.safetensors"),
    ],
    "diffusion_models/lotus-depth-d-v1-1.safetensors": [
        ("jingheya/lotus-depth-d-v1-1", "unet/diffusion_pytorch_model.safetensors"),
    ],
    # BiRefNet
    "background_removal/birefnet.safetensors": [
        ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors"),
    ],
    # Chatterbox (org = ResembleAI)
    "chatterbox/chatterbox/t3_cfg.safetensors": [
        ("ResembleAI/chatterbox", "t3_cfg.safetensors"),
    ],
    "chatterbox/chatterbox/s3gen.safetensors": [
        ("ResembleAI/chatterbox", "s3gen.safetensors"),
    ],
    # ملفات صغيرة (<100MB) مطلوبة في حزمة chatterbox-tts — تُدرَج صراحةً رغم صغرها
    "chatterbox/chatterbox/ve.safetensors": [
        ("ResembleAI/chatterbox", "ve.safetensors"),
    ],
    "chatterbox/chatterbox/conds.pt": [
        ("ResembleAI/chatterbox", "conds.pt"),
    ],
    "chatterbox/chatterbox_multilingual/t3_mtl23ls_v2.safetensors": [
        ("ResembleAI/chatterbox", "t3_mtl23ls_v2.safetensors"),
    ],
    "chatterbox/chatterbox_multilingual/s3gen.pt": [
        ("ResembleAI/chatterbox", "s3gen.pt"),
    ],
    "chatterbox/chatterbox_multilingual/ve.pt": [
        ("ResembleAI/chatterbox", "ve.pt"),
    ],
    "chatterbox/chatterbox_multilingual/conds.pt": [
        ("ResembleAI/chatterbox", "conds.pt"),
    ],
    "chatterbox/chatterbox_turbo/t3_turbo_v1.safetensors": [
        ("ResembleAI/chatterbox-turbo", "t3_turbo_v1.safetensors"),
    ],
    "chatterbox/chatterbox_turbo/s3gen_meanflow.safetensors": [
        ("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors"),
    ],
    "chatterbox/chatterbox_turbo/ve.safetensors": [
        ("ResembleAI/chatterbox-turbo", "ve.safetensors"),
    ],
    "chatterbox/chatterbox_turbo/conds.pt": [
        ("ResembleAI/chatterbox-turbo", "conds.pt"),
    ],
    "chatterbox/chatterbox_vc/s3gen.pt": [
        ("ResembleAI/chatterbox", "s3gen.pt"),
    ],
    "chatterbox/chatterbox_vc/conds.pt": [
        ("ResembleAI/chatterbox", "conds.pt"),
    ],
}

BUNDLES = {
    "qwen-edit-2511": [
        "diffusion_models/qwen-image-edit-2511-Q8_0.gguf",
        "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae/qwen_image_vae.safetensors",
        "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ],
    "qwen-edit-fp8": [
        "diffusion_models/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors",
        "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "vae/qwen_image_vae.safetensors",
        "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    ],
    "flux2-klein": [
        "diffusion_models/flux-2-klein-4b-Q8_0.gguf",
        "text_encoders/qwen_3_4b_fp4_flux2.safetensors",
        "vae/flux2-vae.safetensors",
    ],
    "sd15-classic": [
        "checkpoints/v1-5-pruned-emaonly.safetensors",
        "vae/vae-ft-mse-840000-ema-pruned.safetensors",
    ],
    "chatterbox-tts": [
        "chatterbox/chatterbox/t3_cfg.safetensors",
        "chatterbox/chatterbox/s3gen.safetensors",
        "chatterbox/chatterbox/ve.safetensors",
        "chatterbox/chatterbox/conds.pt",
        "chatterbox/chatterbox_multilingual/t3_mtl23ls_v2.safetensors",
        "chatterbox/chatterbox_multilingual/s3gen.pt",
        "chatterbox/chatterbox_multilingual/ve.pt",
        "chatterbox/chatterbox_multilingual/conds.pt",
        "chatterbox/chatterbox_turbo/t3_turbo_v1.safetensors",
        "chatterbox/chatterbox_turbo/s3gen_meanflow.safetensors",
        "chatterbox/chatterbox_turbo/ve.safetensors",
        "chatterbox/chatterbox_turbo/conds.pt",
        "chatterbox/chatterbox_vc/s3gen.pt",
        "chatterbox/chatterbox_vc/conds.pt",
    ],
    "depth-3d-trellis": [
        "diffusion_models/lotus-depth-d-v1-1.safetensors",
        "clip_vision/dino_v3_L_naf_fp32.safetensors",
        "vae/trellis_2_shape_vae_bf16.safetensors",
        "vae/trellis_2_texture_vae_bf16.safetensors",
    ],
    "extras": [
        "background_removal/birefnet.safetensors",
        "loras/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
        "diffusion_models/pixal3d_multiview_int8_convrot.safetensors",
    ],
}

def scan_local():
    found = {}
    for p in MODELS_DIR.rglob("*"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        if p.suffix.lower() not in EXTS:
            continue
        rel = p.relative_to(MODELS_DIR).as_posix()
        # الملفات المسجّلة صراحةً في CANDIDATES تُفحص مهما كان صغرها
        # (حزمة chatterbox-tts تحتاج ve/conds الصغيرة)؛ غير ذلك: عتبة 100MB
        if p.stat().st_size < MIN_SIZE and rel not in CANDIDATES:
            continue
        found[rel] = p.stat().st_size
    return found

def try_candidate(dest, local_size, cand):
    if isinstance(cand, tuple):
        repo_id, remote_name = cand
    else:
        repo_id, remote_name = cand, Path(dest).name
    full = f"models/{repo_id}/{remote_name}"
    try:
        info = hfs.stat(full)
    except Exception:
        return None
    size = info.get("size") or (info.get("lfs") or {}).get("size") or 0
    if not size:
        return None
    if abs(size - local_size) / local_size <= 0.01:
        return {"repo_id": repo_id, "remote": remote_name, "size": size}
    return {"repo_id": repo_id, "remote": remote_name, "size": size, "size_mismatch": True}

def main():
    local = scan_local()
    print(f"local files >=100MB: {len(local)}")
    verified, orphans, mismatched = {}, [], []
    for dest, size in sorted(local.items()):
        cands = CANDIDATES.get(dest)
        if not cands:
            orphans.append({"dest": dest, "size_gb": round(size / 1e9, 2), "reason": "no-candidates"})
            print(f"[ORPHAN-nocand] {dest}")
            continue
        hit = None
        for cand in cands:
            r = try_candidate(dest, size, cand)
            if r and not r.get("size_mismatch"):
                hit = r
                break
            if r and hit is None:
                hit = r
        if hit is None:
            orphans.append({"dest": dest, "size_gb": round(size / 1e9, 2), "reason": "candidates-failed"})
            print(f"[ORPHAN-fail]   {dest}")
        elif hit.get("size_mismatch"):
            mismatched.append({"dest": dest, "local_gb": round(size / 1e9, 2), **hit})
            print(f"[MISMATCH]      {dest} <- {hit['repo_id']}/{hit['remote']} ({hit['size']/1e9:.2f}GB vs {size/1e9:.2f}GB)")
        else:
            verified[dest] = hit
            print(f"[OK]            {dest} <- {hit['repo_id']}/{hit['remote']}")
    files = []
    for dest, hit in sorted(verified.items()):
        rev = get_revision(hit["repo_id"])
        files.append({
            "dest": dest,
            "source": f"hf://{hit['repo_id']}",
            "remote": hit["remote"],
            "revision": rev,
            "size_gb": round(local[dest] / 1e9, 2),
        })
    for o in orphans:
        if o["dest"] == "diffusion_models/pixal3d_multiview_int8_convrot.safetensors":
            files.append({
                "dest": o["dest"],
                "source": "archive://awAyham/comfyui-models",
                "revision": None,
                "size_gb": o["size_gb"],
            })
    manifest = {
        "version": 1,
        "updated": "2026-10-13",
        "defaults": {"repo_type": "model"},
        "files": files,
        "bundles": BUNDLES,
        "notes": [
            "لا تسحب حزمتين لنموذج واحد في نفس الجلسة (توأم GGUF/fp8)",
            "archive:// يتطلب repo awAyham/comfyui-models على HF",
            "url:// يدعم روابط مباشرة (Civitai) عند الحاجة",
        ],
    }
    # فحص تكامل الحزم: كل dest في أي حزمة يجب أن يكون في files:
    file_dests = {f["dest"] for f in files}
    gaps = sorted({d for bundle in BUNDLES.values() for d in bundle} - file_dests)
    if gaps:
        print("[BUNDLE-GAP] حزم تشير لملفات غير موجودة في files:")
        for g in gaps:
            print(f"  - {g}")
    OUT.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    REPORT.write_text(json.dumps({"verified": len(verified), "orphans": orphans, "mismatched": mismatched}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"verified={len(verified)} orphans={len(orphans)} mismatched={len(mismatched)}")
    if mismatched:
        print("MISMATCHED:", json.dumps(mismatched, indent=2))
    if orphans:
        print("ORPHANS:", json.dumps(orphans, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
