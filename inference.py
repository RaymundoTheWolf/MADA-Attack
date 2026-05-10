import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("Please install huggingface_hub: pip install huggingface-hub")

parser = argparse.ArgumentParser(description="UAP inference script")
parser.add_argument("--images", type=str, required=True, help="input images directory")
parser.add_argument("--output_dir", type=str, required=True, help="output adv images directory")
parser.add_argument("--exts", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"], help="image extensions to process")
args = parser.parse_args()

max_norm = 12 / 255.0
save_quality = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = hf_hub_download(repo_id="AntlersTheWarden/MADA", filename="mada_uap_eps12_imagenet.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
if isinstance(ckpt, dict) and "perturbation" in ckpt:
    perturb = ckpt["perturbation"]
else:
    # support direct tensor saved by torch.save(tensor)
    perturb = ckpt

if not torch.is_tensor(perturb):
    raise TypeError("Loaded perturbation is not a tensor. Please provide a checkpoint with key 'perturbation' or a saved tensor.")

# Move to device for later processing
perturb = perturb.to(device)

# Ensure numeric type float
perturb = perturb.float()

# Clamp loaded perturbation to provided max_norm just in case
perturb = torch.clamp(perturb, -max_norm, max_norm)

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
resize_224 = transforms.Resize((224, 224), interpolation=Image.BICUBIC)

def get_perturbation_for_image(perturb_tensor, target_h=224, target_w=224):
    """
    Return a tensor of shape (3, target_h, target_w) on the same device as perturb_tensor.
    Handles input shapes:
      - (3, H, W)
      - (1, 3, H, W)
      - (N, 3, H, W)  (will take first item)
    """
    p = perturb_tensor
    # normalize dims
    if p.dim() == 4:
        # assume (N, C, H, W) -> take first if N>1
        if p.size(0) > 1:
            p = p[0:1]  # keep batch dim
    elif p.dim() == 3:
        # (C,H,W) -> add batch dim
        p = p.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported perturbation shape: {tuple(p.shape)}")

    # now p is (1, C, H, W)
    # interpolate (must be 4D)
    if (p.size(-2), p.size(-1)) != (target_h, target_w):
        p_resized = F.interpolate(p, size=(target_h, target_w), mode="bilinear", align_corners=False)
    else:
        p_resized = p

    # p_resized is (1, C, target_h, target_w)
    # squeeze batch dim -> (C, target_h, target_w)
    p_resized = p_resized.squeeze(0)

    # final check: C should be 3
    if p_resized.size(0) == 1:
        # maybe grayscale saved; replicate channels
        p_resized = p_resized.repeat(3, 1, 1)
    elif p_resized.size(0) != 3:
        # if extra channels, take first 3
        p_resized = p_resized[:3, :, :]

    # clamp again to max_norm to be safe
    p_resized = torch.clamp(p_resized, -max_norm, max_norm)

    return p_resized.to(device)

perturb_resized = get_perturbation_for_image(perturb, 224, 224)

os.makedirs(args.output_dir, exist_ok=True)
image_files = [f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in args.exts]
image_files = sorted(image_files)

if len(image_files) == 0:
    print(f"[WARN] No images found in {args.images} with extensions {args.exts}")
else:
    print(f"[INFO] Found {len(image_files)} images. Saving adversarial images to {args.output_dir}")

for img_name in tqdm(image_files, desc="Applying 224x224 UAP"):
    img_path = os.path.join(args.images, img_name)
    save_path = os.path.join(args.output_dir, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {img_path}: {e}")
        continue

    # Resize to 224x224 (deterministic)
    img_resized = resize_224(img)
    img_tensor = to_tensor(img_resized).to(device)  # (3,224,224), float in [0,1]

    # perturb_resized is (3,224,224) on device
    # Add perturbation (broadcasting automatically)
    adv_img = img_tensor + perturb_resized
    adv_img = torch.clamp(adv_img, 0.0, 1.0)

    # convert back to PIL and save
    adv_img_pil = to_pil(adv_img.cpu())

    ext = os.path.splitext(img_name)[1].lower()
    try:
        if ext in [".jpg", ".jpeg"]:
            adv_img_pil.save(save_path, quality=save_quality, optimize=True)
        else:
            adv_img_pil.save(save_path, optimize=True)
    except Exception as e:
        # fallback: save without optimization/quality
        adv_img_pil.save(save_path)
        print(f"[WARN] Saving {save_path} with fallback due to: {e}")

print(f"[INFO] Finished. Adversarial images saved to: {args.output_dir}")
