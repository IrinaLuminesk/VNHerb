from torchvision.transforms.v2 import Transform
import torch
import torch.nn.functional as F

class RingMix(Transform):
    def __init__(self, patch_size: int = 16, num_classes: int = None, p: float = 1.0):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.p = p

    def make_params(self, images):
        """Generate parameters for the batch (ring_mask, lambda, etc.)."""
        _, _, h, w = images.shape
        patch_size = self.patch_size
        grid_h, grid_w = h // patch_size, w // patch_size

        cx, cy = (grid_w - 1) / 2.0, (grid_h - 1) / 2.0
        yy, xx = torch.meshgrid(
            torch.arange(grid_h, device=images.device),
            torch.arange(grid_w, device=images.device),
            indexing="ij"
        )
        dist_f = torch.maximum((xx - cx).abs(), (yy - cy).abs())
        ring_idx = torch.floor(dist_f + 1e-6).long()
        ring_mask = (ring_idx % 2 == 1).float()  # 1 = from mixed image
        lam = 1.0 - ring_mask.mean().item()      # fraction of original
        return {"ring_mask": ring_mask, "lam": lam}

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("Expected images of shape [B, C, H, W]")

        params = self.make_params(images)
        params["labels"] = labels
        params["batch_size"] = images.shape[0]

        # Apply the transform logic to images and labels
        mixed_images = self.transform(images, params)
        mixed_labels = self.transform(labels, params)

        return mixed_images, mixed_labels

    def transform(self, inpt: torch.Tensor, params: dict[str, any]):
        if torch.rand(1).item() >= self.p:
            return inpt
        """Core logic: mix images and labels based on the ring_mask."""
        ring_mask = params["ring_mask"]
        lam = params["lam"]
        labels = params["labels"]
        patch_size = self.patch_size
        # 🔹 If input is the labels → mix labels
        if inpt is labels:
            mix_labels = labels.roll(1, 0)

            if labels.ndim == 1:
                if self.num_classes is None:
                    raise ValueError("num_classes must be provided for 1D labels")
                labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
                mix_labels_onehot = F.one_hot(mix_labels, num_classes=self.num_classes).float()
            else:
                labels_onehot = labels.float()
                mix_labels_onehot = mix_labels.float()

            mixed_labels = lam * labels_onehot + (1 - lam) * mix_labels_onehot
            return mixed_labels

        # 🔹 If input is the images → mix using ring pattern
        b, c, h, w = inpt.shape
        mix_img = inpt.roll(1, 0)

        original_patches = inpt.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        mix_patches = mix_img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        replace = ring_mask[None, None, :, :, None, None].bool()
        mixed_patches = torch.where(replace, mix_patches, original_patches)
        mixed = mixed_patches.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)
        return mixed