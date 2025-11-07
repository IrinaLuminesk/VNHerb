
# from torchvision.transforms.v2 import Transform
# import torch
# import torch.nn.functional as F

# class RingMix(Transform):
#     def __init__(self, patch_size: int = 16, num_classes: int = None, p: float = 1.0):
#         """
#         RingMix transform: mixes patches of images in a ring pattern and
#         mixes labels for classification or segmentation.

#         Args:
#             patch_size (int): Size of the square patch for mixing.
#             num_classes (int, optional): Required for 1D classification labels.
#             p (float): Probability of applying the transform.
#         """
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_classes = num_classes
#         self.p = p

#     def make_params(self, images: torch.Tensor):
#         """Generate ring_mask and lam for the batch."""
#         _, _, h, w = images.shape
#         patch_size = self.patch_size
#         grid_h, grid_w = h // patch_size, w // patch_size

#         cx, cy = (grid_w - 1) / 2.0, (grid_h - 1) / 2.0
#         yy, xx = torch.meshgrid(
#             torch.arange(grid_h, device=images.device),
#             torch.arange(grid_w, device=images.device),
#             indexing="ij"
#         )
#         dist_f = torch.maximum((xx - cx).abs(), (yy - cy).abs())
#         ring_idx = torch.floor(dist_f + 1e-6).long()
#         ring_mask = (ring_idx % 2 == 1).float()
#         lam = 1.0 - ring_mask.mean().item()
#         return {"ring_mask": ring_mask, "lam": lam}

#     def forward(self, images: torch.Tensor, labels: torch.Tensor):
#         if images.ndim != 4:
#             raise ValueError("Expected images of shape [B, C, H, W]")

#         params = self.make_params(images)
#         params["labels"] = labels
#         params["batch_size"] = images.shape[0]

#         # Apply transform separately for images and labels
#         mixed_images = self.transform_images(images, params)
#         mixed_labels = self.transform_labels(labels, params)

#         return mixed_images, mixed_labels

#     def transform_images(self, images: torch.Tensor, params: dict):
#         """Mix images using ring pattern."""
#         if torch.rand(1).item() >= self.p:
#             return images  # skip transform

#         ring_mask = params["ring_mask"]
#         patch_size = self.patch_size
#         b, c, h, w = images.shape
#         mix_img = images.roll(1, 0)

#         original_patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         mix_patches = mix_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

#         replace = ring_mask[None, None, :, :, None, None].bool()
#         mixed_patches = torch.where(replace, mix_patches, original_patches)
#         mixed = mixed_patches.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
#         return mixed

#     def transform_labels(self, labels: torch.Tensor, params: dict):
#         """Mix labels safely for classification (1D) or soft labels/masks."""
#         lam = params["lam"]
#         mix_labels = labels.roll(1, 0)

#         # 1D classification labels
#         if labels.ndim == 1:
#             if self.num_classes is None:
#                 raise ValueError("num_classes must be provided for 1D labels")
#             labels_onehot = F.one_hot(labels.long(), num_classes=self.num_classes).float()
#             mix_labels_onehot = F.one_hot(mix_labels.long(), num_classes=self.num_classes).float()

#             if torch.rand(1).item() >= self.p:
#                 return labels_onehot  # ensure correct shape even if skipped
#             mixed_labels = lam * labels_onehot + (1 - lam) * mix_labels_onehot
#             return mixed_labels

#         # Soft labels or segmentation masks
#         else:
#             labels = labels.float()
#             mix_labels = mix_labels.float()
#             if torch.rand(1).item() >= self.p:
#                 return labels  # skip mixing
#             mixed_labels = lam * labels + (1 - lam) * mix_labels
#             return mixed_labels
