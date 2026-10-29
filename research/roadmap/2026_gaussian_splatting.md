# Survey: Neural Network Architectures for Gaussian Splatting

## 1. Scope and framing

Vanilla 3D Gaussian Splatting (3DGS, Kerbl et al., SIGGRAPH 2023) is not a neural network. It is an explicit set of anisotropic Gaussians (mean, scale + quaternion covariance, opacity, spherical-harmonic color) fit per scene by gradient descent through a differentiable tile rasterizer. Neural networks enter the pipeline in five distinct roles:

| Role | What the network does | Representative work |
|---|---|---|
| Neural-field hybrid | Decodes Gaussian attributes from stored features during per-scene optimization | Scaffold-GS, Octree-GS, HAC |
| Deformation / motion | Maps canonical Gaussians to time t | Deformable 3DGS, 4DGaussians, SC-GS |
| Feed-forward reconstruction | Predicts all Gaussians from images in one pass | pixelSplat, MVSplat, GS-LRM, AnySplat |
| Generative | Samples Gaussians (or latents decoded to Gaussians) | LGM, GaussianCube, TRELLIS, DiffSplat |
| Auxiliary heads | Semantics, compression entropy models, avatars | LangSplat, HAC, GaussianAvatars |

## 2. Neural-field hybrids (per-scene optimization)

**Pattern:** store learnable features on a sparse structure, decode Gaussian attributes with tiny MLPs at render time.

| Method | Structure | Decoder | Purpose |
|---|---|---|---|
| Scaffold-GS (CVPR 2024) | Voxel anchors with feature vectors | MLPs emit k "neural Gaussians" per anchor, conditioned on view direction/distance | Less redundancy, view adaptivity |
| Octree-GS | Multi-level octree of anchors | Same as Scaffold-GS | Level-of-detail, large scenes |
| HAC / HAC++ (ECCV 2024) | Anchors + hash-grid context | Entropy model over anchor attributes | Compression (~75x+) |
| Compact-3DGS (Lee et al.) | Hash grid | MLP color field replacing SH | Memory |
| GS-W, WildGaussians | Per-image appearance embeddings | MLP modulating color | In-the-wild photo collections |

**Trade-off:** MLP evaluation per frame costs FPS; buys memory, generalization across view conditions, and compressibility.

## 3. Dynamic / 4D architectures

| Family | Architecture | Examples |
|---|---|---|
| Deformation MLP | Canonical Gaussians + MLP(x, t) -> delta position/rotation/scale | Deformable 3D Gaussians (CVPR 2024) |
| Factorized plane encoder + heads | HexPlane-style 4D feature planes, small MLP heads | 4DGaussians (Wu et al., CVPR 2024) |
| Sparse control points | MLP predicts 6-DoF trajectories of control nodes, LBS to Gaussians | SC-GS (CVPR 2024) |
| Explicit spacetime primitives | 4D Gaussians sliced at t, or polynomial motion + feature splatting with small color decoder | Real-time 4DGS (ICLR 2024), Spacetime Gaussians |
| Motion bases / scaffolds | Low-rank SE(3) motion bases fit with depth/track priors | Shape of Motion, MoSca |

Feed-forward 4D is now an active branch. 4DGT trains a feed-forward Transformer to directly predict consistent 4D Gaussians from monocular videos; other entries include DGS-LRM and MoVieS. Driving is a major application: ReconDrive maps inputs to a 4D Gaussian Splatting representation that separates static geometry from dynamic traffic participants, and DGGT reconstructs dynamic driving scenes from unposed images.

## 4. Feed-forward (generalizable) reconstruction

This is the dominant architecture research area. Four generations:

### 4.1 Pixel-aligned with explicit geometry reasoning

Splatter Image showed single-view object reconstruction with Gaussian maps, while pixelSplat predicts 3D Gaussians from posed image pairs by lifting pixel-aligned predictions along camera rays. MVSplat improved on this with a plane-sweep cost volume, and DepthSplat links multi-view depth estimation to Gaussian prediction using depth priors.

| Method | Backbone | Depth mechanism | Notes |
|---|---|---|---|
| Splatter Image | U-Net | Direct regression | 1 view, objects |
| pixelSplat (CVPR 2024) | ResNet/ViT + epipolar transformer | Probabilistic depth bins, reparameterized sampling | 2 posed views |
| MVSplat (ECCV 2024) | CNN + cross-view transformer + cost volume + U-Net refiner | Plane sweep | ~10x fewer params than pixelSplat |
| DepthSplat (CVPR 2025) | Cost volume + monocular depth (Depth Anything) branch | Fused | Scales to more views |
| MVSplat360 | MVSplat + video diffusion | Plane sweep | 360 scenes |

### 4.2 Large Reconstruction Models (transformer-scale)

| Method | Tokenization | Backbone | Output |
|---|---|---|---|
| LGM (ECCV 2024) | 4 multi-view images (from MV diffusion) + Plucker rays | Asymmetric U-Net with cross-view attention | Per-pixel Gaussians |
| GS-LRM (ECCV 2024) | Image patches + Plucker ray embeddings | Plain transformer, linear unpatchify | Per-pixel Gaussians |
| GRM (ECCV 2024) | Same idea | Transformer + upsampler | Pixel-aligned |
| Long-LRM | 32 views at ~960x540 | Mamba2 + transformer hybrid, token merging | Scene-scale |
| iLRM | Viewpoint tokens decoupled from inputs | Transformer | Low-res Gaussian grid |

iLRM decouples viewpoint tokens from the input images and predicts Gaussians in a separate low-resolution grid, giving a more compact set, but such uniform downsampling reduces primitives everywhere regardless of scene complexity.

### 4.3 Pose-free, foundation-model-backed

Built on DUSt3R / MASt3R / VGGT style geometry transformers that regress pointmaps and cameras.

| Method | Backbone | Key idea |
|---|---|---|
| Splatt3R | MASt3R + Gaussian head | Stereo pointmaps to Gaussians |
| NoPoSplat (ICLR 2025) | CroCo/DUSt3R-initialized ViT | Predict Gaussians in canonical frame of first view |
| FLARE (CVPR 2025) | Cascaded pose then geometry then appearance | Pose as bridge |
| AnySplat (TOG 2025) | VGGT-style alternating attention | Joint Gaussians + intrinsics/extrinsics, voxelized |
| YoNoSplat, Depth Anything 3 | Geometry foundation models with GS heads | Unified pose-free/posed |

AnySplat is the reference design for this generation. Its geometry transformer follows VGGT: each image is patchified into tokens via a pretrained DINOv2 backbone with patch size 14 and token dimension 1024. It uses a 24-layer alternating-attention transformer partially initialized from VGGT, about 886M parameters total. The pipeline comprises the geometry transformer, camera pose prediction, pixel-wise Gaussian parameter prediction, and a differentiable voxelization module. Pose regularization ties predicted cameras to pseudo ground truth from VGGT, i.e. distillation. It scales to hundreds of input views and reconstructs scenes in seconds, versus up to 10 minutes for optimization-based dense-view pipelines.

### 4.4 Beyond pixel alignment (2025-2026 frontier)

The core critique: most methods map each 2D pixel to a 3D Gaussian, tightly coupling the Gaussian count to the input images. This makes reconstructions depend on the number of input views, produces view-biased density, and introduces alignment errors under occlusion or low texture.

| Method | Venue | Anchoring | Architecture |
|---|---|---|---|
| VolSplat | ECCV 2026 | Voxel-aligned | Sparse 3D decoder refines features in 3D and predicts one Gaussian per occupied voxel |
| AnchorSplat | CVPR 2026 | 3D anchors | Two-stage: Gaussian decoder predicts anchor-aligned Gaussians, then a Gaussian refiner improves quality and view consistency |
| SparseSplat | CVPR 2026 | Pixel-unaligned, adaptive | Adjusts density to scene content; SOTA on DL3DV with 150k vs DepthSplat's 688k Gaussians |
| ATSplat | 2026 | Adaptive 3D tokens | Lifts patch-level depth and camera cues into sparse 3D anchor tokens; over 5.7x fewer Gaussians than dense methods on RE10K and DL3DV |
| Z-Order Transformer | CVPR 2026 oral | Serialized Gaussian set | Orders unstructured Gaussians along a Z-order curve so sparse attention can model context and compress primitives in one pass |
| VG²GT | 2026 | Voxel-Gaussian | Regresses Gaussians and camera parameters from arbitrary numbers of unposed images |

Dense-view input scaling is a parallel thread: ZPressor compresses multi-view inputs into a compact latent state, letting feed-forward 3DGS models scale past 100 input views.

## 5. Generative architectures

| Paradigm | Architecture | Examples |
|---|---|---|
| Score distillation | Per-asset optimization guided by 2D diffusion (no new network) | DreamGaussian, GaussianDreamer |
| Multi-view diffusion -> LRM | MV diffusion generates views, feed-forward LRM lifts to Gaussians | LGM, GRM, InstantMesh-style |
| Native 3D latent diffusion | Gaussians organized into structured grid/latent, 3D U-Net or DiT denoiser | GaussianCube (optimal-transport voxel fit), L3DG |
| Structured latents, multi-decoder | Sparse voxel latent + rectified-flow transformer, decoded to Gaussians / radiance field / mesh | TRELLIS (CVPR 2025) |
| Repurposed image diffusion | Splatter-image latents generated by fine-tuned 2D diffusion | DiffSplat (ICLR 2025) |
| Scene / world generation | Video or 4D generators with Gaussian output | WorldSplat, WonderWorld |
| Avatars | Parametric-model-rigged Gaussians; feed-forward human Gaussians | GaussianAvatars (FLAME), GPS-Gaussian, LHM |

## 6. Auxiliary / task-specific heads

| Task | Mechanism | Examples |
|---|---|---|
| Open-vocabulary semantics | Per-Gaussian low-dim features; autoencoder to/from CLIP space | LangSplat, Feature 3DGS, Gaussian Grouping |
| Compression | Context/entropy models (hash grid, triplane), feed-forward codecs | HAC, CAT-3DGS, FCGS, D-FCGS, GenSplatCodec |
| SLAM | Mostly optimization; networks for depth/pose priors | SplaTAM, MonoGS, GS-SLAM |
| Renderer variants (constrain network outputs) | 2D surfels, ray-traced/unscented-transform Gaussians, anti-aliasing filters | 2DGS, 3DGRT, 3DGUT, Mip-Splatting |

## 7. Cross-cutting design dimensions

| Dimension | Options (oldest -> newest) |
|---|---|
| Primitive anchoring | Per-pixel -> voxel -> 3D anchor / adaptive token -> free set with serialization |
| Backbone | CNN U-Net -> epipolar transformer -> cost volume + U-Net -> plain ViT (LRM) -> alternating-attention geometry transformer (VGGT) -> sparse 3D conv / sparse attention |
| Camera input | Posed (Plucker rays) -> pose-free with predicted intrinsics/extrinsics |
| Supervision | Photometric render loss (L2 + LPIPS) -> + depth/pose distillation from foundation models -> self-supervised (no 3D GT) |
| View count | 1-2 -> 4-32 -> 100s via token compression / linear-time sequence models |
| Training data | Objaverse (objects), RealEstate10K, ACID, DL3DV, ScanNet (scenes), Waymo/nuScenes (driving) |

## 8. Open problems

1. **Primitive budget:** decoupling Gaussian count from pixel count while retaining detail in complex regions.
2. **Dense-view scaling:** quadratic attention over N views x tokens; candidates are latent compression, Mamba/linear attention, streaming state.
3. **Geometry vs appearance:** feed-forward Gaussians render well but give poor surfaces; integration with 2DGS-style surfels is immature.
4. **Pose-free evaluation:** TokenSplat's authors observe that AnySplat's scenes show distortions that only become obvious when test views are aligned with ground truth following the NoPoSplat protocol, so benchmark protocols matter.
5. **Dynamics in one pass:** feed-forward 4D from monocular, unposed, long video.
6. **Unifying reconstruction and generation:** single models that reconstruct observed regions and hallucinate unobserved ones consistently.

## 9. Suggested reading order

1. 3DGS (Kerbl 2023), 2DGS, Mip-Splatting
2. Scaffold-GS, HAC
3. Deformable 3DGS, 4DGaussians, SC-GS
4. pixelSplat, MVSplat, DepthSplat
5. LGM, GS-LRM, Long-LRM
6. DUSt3R, VGGT, NoPoSplat, AnySplat
7. VolSplat, AnchorSplat, SparseSplat, ATSplat, Z-Order Transformer
8. GaussianCube, TRELLIS, DiffSplat
9. 4DGT, DGS-LRM, ReconDrive