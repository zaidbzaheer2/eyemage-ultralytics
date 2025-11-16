# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# ultralytics/nn/modules/physics_detect.py
import torch
import torch.nn as nn

from ...utils.tal import dist2bbox, make_anchors
from .block import DFL
from .conv import Conv, DWConv


class PhysicsDetect(nn.Module):
    """YOLO Detect head with physics-aware features for wet surface detection."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4

        self.stride = torch.zeros(self.nl)

        # Standard detection branches
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        # Match original cv3 exactly
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        # PHYSICS BRANCHES - Core of our innovation
        self.cv_specular = nn.ModuleList(
            nn.Sequential(
                Conv(x, max(64, x // 4), 3),
                nn.Conv2d(max(64, x // 4), 1, 1),  # Specular intensity
                nn.Sigmoid(),  # 0-1 range
            )
            for x in ch
        )

        self.cv_smoothness = nn.ModuleList(
            nn.Sequential(
                Conv(x, max(64, x // 4), 3),
                nn.Conv2d(max(64, x // 4), 1, 1),  # Surface smoothness
                nn.Sigmoid(),  # 0-1 range
            )
            for x in ch
        )

        self.cv_reflectance = nn.ModuleList(
            nn.Sequential(
                Conv(x, max(64, x // 4), 3),
                nn.Conv2d(max(64, x // 4), 1, 1),  # Reflectance ratio
                nn.Sigmoid(),  # 0-1 range
            )
            for x in ch
        )

        # Fusion layer to combine physics features with classification
        self.fusion_conv = nn.ModuleList(
            nn.Sequential(
                Conv(self.nc + 3, 256, 1),  # +3 for physics features
                Conv(256, 128, 3),
                nn.Conv2d(128, self.nc, 1),  # Back to original class count
            )
            for _ in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Forward pass with physics feature fusion."""
        physics_outputs = []

        for i in range(self.nl):
            # Extract physics features
            specular = self.cv_specular[i](x[i])  # [B, 1, H, W]
            smoothness = self.cv_smoothness[i](x[i])  # [B, 1, H, W]
            reflectance = self.cv_reflectance[i](x[i])  # [B, 1, H, W]

            # Standard detection features
            reg_feat = self.cv2[i](x[i])  # [B, 64, H, W]
            cls_feat = self.cv3[i](x[i])  # [B, nc, H, W]

            # FUSE PHYSICS WITH CLASSIFICATION
            # Concatenate physics features with class features
            physics_features = torch.cat([specular, smoothness, reflectance], dim=1)  # [B, 3, H, W]
            enhanced_cls = torch.cat([cls_feat, physics_features], dim=1)  # [B, nc+3, H, W]

            # Fusion to get physics-aware class predictions
            fused_cls = self.fusion_conv[i](enhanced_cls)  # [B, nc, H, W]

            # Store physics features for loss computation
            physics_outputs.append(
                {
                    "specular": specular,
                    "smoothness": smoothness,
                    "reflectance": reflectance,
                    "combined_physics": physics_features,
                }
            )

            # Final output (regression + fused classification)
            x[i] = torch.cat([reg_feat, fused_cls], dim=1)

        if self.training:
            return x, physics_outputs  # Return both detection and physics outputs for loss

        # Inference path
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not getattr(self, "end2end", False) and not self.xyxy,
            dim=1,
        )

    def bias_init(self):
        """Initialize biases - enhanced for wet surface detection."""
        # Boost wet class confidence initialization
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: self.nc] = torch.log(torch.tensor(5 / self.nc / (640 / s) ** 2))

            # Optional: Slightly boost wet class initialization if you know its index
            # Example: if wet surface is class 0
            # b[-1].bias.data[0] = torch.log(torch.tensor(8 / self.nc / (640 / s) ** 2))

    def get_physics_features(self, x):
        """Extract only physics features for analysis or visualization."""
        physics_maps = []
        for i in range(self.nl):
            specular = self.cv_specular[i](x[i])
            smoothness = self.cv_smoothness[i](x[i])
            reflectance = self.cv_reflectance[i](x[i])

            physics_maps.append(
                {"specular": specular, "smoothness": smoothness, "reflectance": reflectance, "layer": i}
            )
        return physics_maps
