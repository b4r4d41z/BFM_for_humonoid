"""Shared frozen DINOv2, camera-specific projections, and compact MLP fusion."""
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import torch
from torch import nn
from torch.nn import functional as F
from bfm.contracts.vision import CAMERAS, PROFILE, PROPRIO_DIM, PROPRIO_JOINTS
from .networks import mlp


@dataclass
class VisionConfig:
    backbone: str = 'dinov2'
    backbone_id: str = 'facebook/dinov2-small'
    revision: str = 'main'  # Resolved to the downloaded immutable revision on first load.
    backbone_config: dict = field(default_factory=dict)
    weights_sha256: str = ''
    implementation_version: str = ''
    attention_implementation: str = 'eager'
    preprocessing_torch_version: str = ''
    frozen: bool = True
    cameras: list = field(default_factory=lambda: list(CAMERAS))
    proprio_joints: list = field(default_factory=lambda: list(PROPRIO_JOINTS))
    proprio_dim: int = PROPRIO_DIM
    image_size: int = 518  # Official DINOv2 pretrained input, divisible by patch size 14.
    preprocessing: str = 'rgb_letterbox_bicubic_antialias_imagenet_v1'
    mean: list = field(default_factory=lambda: [.485, .456, .406])
    std: list = field(default_factory=lambda: [.229, .224, .225])
    feature_dim: int = 384
    feature_batch_size: int = 16
    projection_dim: int = 64
    fusion: str = 'camera_projections_concat_mlp_v1'
    embedding_dim: int = 256

    def __post_init__(self):
        if self.backbone not in ('dinov2', 'mock') or not self.frozen:
            raise ValueError('Use frozen dinov2 (production) or explicit mock (tests only)')
        if self.attention_implementation != 'eager':
            raise ValueError('This profile pins eager attention for train/restore consistency')
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or any(c not in CAMERAS for c in self.cameras):
            raise ValueError('Unknown/duplicate camera identity')
        if self.proprio_joints != list(PROPRIO_JOINTS) or self.proprio_dim != PROPRIO_DIM:
            raise ValueError('vision_proprio_v1 proprioception ordering/dimension mismatch')
        if self.preprocessing != 'rgb_letterbox_bicubic_antialias_imagenet_v1' or self.mean != [.485, .456, .406] or self.std != [.229, .224, .225]:
            raise ValueError('Unsupported DINOv2 preprocessing identity')
        if self.fusion != 'camera_projections_concat_mlp_v1':
            raise ValueError('Unsupported fusion architecture')
        if min(self.image_size, self.feature_dim, self.feature_batch_size, self.projection_dim, self.embedding_dim) < 1:
            raise ValueError('Vision dimensions must be positive')
        if self.backbone == 'dinov2' and self.image_size % 14:
            raise ValueError('DINOv2 image_size must be divisible by patch size 14')


def preprocess_rgb(images, cfg, device='cpu'):
    """RGB uint8 NHWC -> normalized NCHW; resize whole frame, mean-color padding."""
    x = torch.as_tensor(images, device=device)
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[-1] != 3 or min(x.shape[1:3]) < 1:
        raise ValueError('Expected uint8 RGB [N,H,W,3]')
    x = x.permute(0, 3, 1, 2).float() / 255.
    scale = cfg.image_size / max(x.shape[-2:])
    height, width = [max(1, min(cfg.image_size, round(n*scale))) for n in x.shape[-2:]]
    x = F.interpolate(x, size=(height, width), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    mean = x.new_tensor(cfg.mean)[None, :, None, None]
    std = x.new_tensor(cfg.std)[None, :, None, None]
    x = (x-mean)/std
    dh, dw = cfg.image_size-height, cfg.image_size-width
    return F.pad(x, (dw//2, dw-dw//2, dh//2, dh-dh//2))


def weights_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MockBackbone(nn.Module):
    """Deterministic parameter-free test fixture; never a production fallback."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        values = torch.cat((x.mean((-1, -2)), x.square().mean((-1, -2))), -1)
        return values.repeat(1, (self.dim+5)//6)[:, :self.dim]


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.camera_projections = nn.ModuleDict({c: nn.Sequential(nn.Linear(cfg.feature_dim, cfg.projection_dim), nn.Tanh()) for c in cfg.cameras})
        self.proprio = mlp(cfg.proprio_dim, cfg.projection_dim, cfg.projection_dim, 1)
        self.fuse = mlp((len(cfg.cameras)+1)*cfg.projection_dim, cfg.embedding_dim, cfg.embedding_dim, 1)
        self.cameras = cfg.cameras

    def forward(self, proprio, features):
        return self.fuse(torch.cat([self.camera_projections[c](features[c]) for c in self.cameras] + [self.proprio(proprio)], -1))


class ObservationEncoder(nn.Module):
    def __init__(self, config, *, initialize_pretrained=True):
        super().__init__()
        self.cfg = VisionConfig(**config)
        cfg = self.cfg
        torch_version = str(torch.__version__)
        if initialize_pretrained:
            cfg.preprocessing_torch_version = torch_version
        elif cfg.preprocessing_torch_version.split('+')[0] != torch_version.split('+')[0]:
            raise ValueError(f'Checkpoint preprocessing requires torch {cfg.preprocessing_torch_version}, installed {torch_version}')
        if cfg.backbone == 'mock':
            self.backbone = MockBackbone(cfg.feature_dim)
            cfg.implementation_version = 'mock_v1'
        else:
            try:
                from transformers import Dinov2Config, Dinov2Model
            except ImportError as exc:
                raise ImportError('Install requirements-vision.txt for frozen pretrained DINOv2') from exc
            version = importlib.metadata.version('transformers')
            if not initialize_pretrained and cfg.implementation_version != version:
                raise ValueError(f'Checkpoint requires transformers {cfg.implementation_version}, installed {version}')
            if initialize_pretrained:
                self.backbone = Dinov2Model.from_pretrained(cfg.backbone_id, revision=cfg.revision,
                                                          attn_implementation=cfg.attention_implementation)
                cfg.backbone_config = self.backbone.config.to_dict()
                cfg.revision = getattr(self.backbone.config, '_commit_hash', None) or cfg.revision
                cfg.feature_dim = self.backbone.config.hidden_size
                cfg.implementation_version = version
            else:
                if not cfg.backbone_config or not cfg.weights_sha256:
                    raise ValueError('Checkpoint missing frozen backbone architecture/weights identity')
                architecture = Dinov2Config.from_dict(cfg.backbone_config)
                architecture._attn_implementation = cfg.attention_implementation
                self.backbone = Dinov2Model(architecture)
                if self.backbone.config.hidden_size != cfg.feature_dim:
                    raise ValueError('Checkpoint backbone feature dimension mismatch')
        self.backbone.eval().requires_grad_(False)
        if initialize_pretrained:
            cfg.weights_sha256 = weights_digest(self.backbone)
        self.fusion = Fusion(cfg)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def features(self, observation):
        # Separate frozen feature API permits a fingerprinted derived cache later.
        device = next(self.fusion.parameters()).device
        batch = len(observation['proprio'])
        if batch < 1 or any(len(observation[c]) != batch for c in self.cfg.cameras):
            raise ValueError('Every camera must have the same nonzero batch size as proprioception')
        pixels = torch.cat([preprocess_rgb(observation[c], self.cfg, device) for c in self.cfg.cameras])
        chunks = []
        for pixel_batch in pixels.split(self.cfg.feature_batch_size):
            output = self.backbone(pixel_batch)
            chunks.append(output if self.cfg.backbone == 'mock' else output.last_hidden_state[:, 0])
        features = torch.cat(chunks)
        if features.shape != (batch*len(self.cfg.cameras), self.cfg.feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Invalid frozen backbone features')
        return dict(zip(self.cfg.cameras, features.split(batch)))

    def forward(self, observation, normalized_proprio, *, fusion=None):
        return (self.fusion if fusion is None else fusion)(normalized_proprio, self.features(observation))

    def identity(self):
        return {'observation_profile': PROFILE, **asdict(self.cfg)}
