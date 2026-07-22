from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from transformers import T5EncoderModel, T5Tokenizer

from genie.modules.blocks import patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer, VectorQuantizer, \
                                                     MVSpatioTemporalTransformer, MVSpatioTransformer


from torchvision import transforms
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


import os

# Bundled offline assets (DINOv2 code + weights, T5-base). Points at the repo's
# ``assets/`` directory so training needs no network / no ~/.cache priming.
# Override with the ``UNIVLA_ASSETS_DIR`` env var if assets live elsewhere.
_DEFAULT_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
)
ASSETS_DIR = os.environ.get("UNIVLA_ASSETS_DIR", _DEFAULT_ASSETS_DIR)

DINOV2_REPO_DIR = os.path.join(ASSETS_DIR, "dinov2", "facebookresearch_dinov2_main")
DINOV2_WEIGHTS = os.path.join(ASSETS_DIR, "dinov2", "dinov2_vitb14_reg4_pretrain.pth")
T5_BASE_DIR = os.path.join(ASSETS_DIR, "t5-base")


def load_dino_encoder(model_name: str = "dinov2_vitb14_reg"):
    """Load a DINOv2 encoder fully offline from bundled ``assets/``.

    Loads the model definition from ``assets/dinov2/facebookresearch_dinov2_main``
    with ``source='local'`` (no ``api.github.com`` check), builds it with
    ``pretrained=False``, then loads the bundled weights directly — so nothing is
    ever fetched from the network. Raises if the assets are missing.
    """
    if not os.path.isdir(DINOV2_REPO_DIR):
        raise FileNotFoundError(
            f"DINOv2 code not found at {DINOV2_REPO_DIR}. See README 'offline assets'."
        )
    if not os.path.isfile(DINOV2_WEIGHTS):
        raise FileNotFoundError(
            f"DINOv2 weights not found at {DINOV2_WEIGHTS}. See README 'offline assets'."
        )
    model = torch.hub.load(DINOV2_REPO_DIR, model_name, source="local", pretrained=False)
    state_dict = torch.load(DINOV2_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model





class UncontrolledDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(UncontrolledDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = load_dino_encoder('dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=num_latents,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Load T5 text encoder model
        self.text_encoder = T5EncoderModel.from_pretrained(T5_BASE_DIR, local_files_only=True)
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(768, model_dim)

        # Load T5 tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(T5_BASE_DIR, local_files_only=True)

    def encode_text(self, lang: List):
        # Tokenize the batch with padding to the longest sequence
        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 

        # Access the input IDs and attention masks
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Access the last hidden states
        last_hidden_states = encoder_outputs.last_hidden_state

        return last_hidden_states, attention_mask

    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        # DINOv2 is frozen (requires_grad_(False)); run under no_grad so autograd
        # doesn't retain its activations — ~13% faster on this stage + less GPU mem.
        with torch.no_grad():
            dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)

        # Encode
        z = self.encoder(padded_patches, lang_embed, attention_mask)
      
        # Get latent action for all future frames
        z = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z = z.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq(z)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)
        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "indices": indices,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
        lang_embed = self.lang_proj(lang_embed)
        attention_mask = torch.cat([torch.ones((B, self.num_codes + (H // self.patch_size)**2)).to(self.device),
                                    attention_mask],
                                    dim = -1)

        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_q"])
        video_action_patches = torch.cat([action_patches, video_patches], dim=2)

        # Decode
        video_recon = self.decoder(video_action_patches, lang_embed.unsqueeze(1), attention_mask)
        video_recon = video_recon[:, :, self.num_codes: self.num_codes + video_patches.shape[2]] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device




class ControllableDINOLatentActionModel(nn.Module):
    """
    Latent action VQ-VAE.
    """

    def __init__(
            self,
            in_dim: int,
            model_dim: int,
            latent_dim: int,
            num_latents: int,
            patch_size: int,
            enc_blocks: int,
            dec_blocks: int,
            num_heads: int,
            dropout: float = 0.0
    ) -> None:
        super(ControllableDINOLatentActionModel, self).__init__()
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size ** 2

        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        self.dino_encoder = load_dino_encoder('dinov2_vitb14_reg')
        self.dino_encoder.requires_grad_(False)

        dino_dim = 768

        self.num_codes = 4
        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
        nn.init.uniform_(self.action_latent, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=dino_dim,
            model_dim=model_dim,
            out_dim=latent_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
            causal_temporal=True,
            to_out=False,
        )

        self.to_codebook = nn.Linear(model_dim, latent_dim)
        self.to_codebook_uncontrol = nn.Linear(model_dim, latent_dim)
        self.vq = VectorQuantizer(
            num_latents=16,
            latent_dim=latent_dim,
            code_restart=True,
        )
        ## Decoder: Spatial Transformer
        self.patch_up = nn.Linear(dino_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.action_up_uncontrol = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=dino_dim,        # Dim of DINOv2-Base
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.vq_action = VectorQuantizer(
                num_latents=num_latents,
                latent_dim=latent_dim,
                code_restart=True,
            )
        self.action_latent_controllable = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))
        nn.init.uniform_(self.action_latent_controllable, a=-1, b=1)

        # we only optimize the new tack-centric codebook in stage-2
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        # DINOv2 is frozen (requires_grad_(False)); run under no_grad so autograd
        # doesn't retain its activations — ~13% faster on this stage + less GPU mem.
        with torch.no_grad():
            dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_uncontrol": indices_uncontrol,
        }

    def forward(self, batch: Dict) -> Dict:
        # Encode + VQ
        B, T = batch["videos"].shape[:2]
        H, W = batch["videos"].shape[3:5]

        outputs = self.vq_encode(batch["videos"]) 
        video_patches = self.patch_up(outputs["patches"][:, :-1])

        # Decode
        video_action_patches = torch.cat([self.action_up(outputs["z_q"]), 
                                          self.action_up_uncontrol(outputs["z_q_uncontrol"]), 
                                          video_patches],
                                          dim=2)
        video_recon = self.decoder(video_action_patches)
        video_recon = video_recon[:, :, -video_patches.shape[2]:] 

        outputs.update(
            {
                "recon": video_recon,
                "target": outputs["patches"][:, [-1]]
            }
        )
        return outputs

    @property
    def device(self):
        return next(self.parameters()).device
