# SplatSORT: Belief-Splat Multi-Object Tracking

High-level system design · Nov 11, 2026 · @Nikolas Markou

## Overview

SplatSORT is an online multi-object tracker in which every track is a small set of 3D Gaussians, and association is scored by rendering the tracker's own uncertainty.

**Problem.** Tracking-by-detection systems in the SORT and DeepSORT family store each track as a 2D box, a Kalman state and an appearance vector. They fail in three recurring ways: occluded tracks age out or swap identity, appearance vectors drift when objects turn, and look-alike objects cannot be told apart. All three are 3D problems being solved in 2D.

**Core idea: belief splats.** A track's shape and appearance are Gaussians, and the Kalman filter's pose uncertainty is also a Gaussian. The two combine in closed form, so the renderer can draw the expected object under the filter's posterior. Rendering all tracks together gives an occlusion-aware, per-pixel likelihood of where each track should be visible.

**Building blocks.** Four existing components are coupled, not chained:

- **Depth Anything (V2 or 3):** metric depth to lift detections into 3D and to order tracks front to back.
- **3D Gaussian Splatting:** the object representation and the differentiable renderer.
- **LightGlue:** viewpoint-robust geometric verification between a rendered track and a detection crop.
- **DeepSORT-style tracking:** Kalman filtering, gating, Hungarian assignment and track lifecycle.

**Thesis.** The contribution is the coupling: the renderer becomes the tracker's measurement model, and the tracker's covariance becomes part of what is rendered.

## Goals, non-goals and success criteria

The system targets fewer identity switches under occlusion and viewpoint change, from a single moving RGB camera, online.

**Goals**

- Online, causal tracking of generic object classes from monocular RGB video.
- Occlusion-aware association: an occluded track is explained by its occluder, not aged out.
- Viewpoint-robust re-identification without a class-specific ReID network.
- A drop-in replacement for the association stage of existing trackers, keeping their detectors.

**Non-goals**

- Photorealistic novel-view synthesis of tracked objects. Gaussians only need to be good enough to associate.
- Full dynamic scene reconstruction or background mapping.
- Requiring RGB-D, lidar, multi-camera rigs or known camera poses.
- Training a new detector.

**Success criteria**

| Criterion | Measure | Pass condition |
| --- | --- | --- |
| Association quality | AssA, IDF1 on DanceTrack val | Above BoT-SORT and OC-SORT with identical detections |
| Occlusion robustness | ID switches on MOT17 boxes with annotated visibility below 50% | Lower than best baseline |
| No regression on detection | DetA | Within noise of baseline |
| Ablation validity | Belief splat vs point-estimate render | Belief splat wins on AssA |
| Throughput | Frames per second at 1080p, single GPU | Target set after Phase 1 profiling |

## Positioning and novelty claim

Each building block has been paired with Gaussians before; rendering the filter posterior as the association likelihood for online monocular multi-object tracking has not, as far as a targeted search shows.

| Work | What it does | What SplatSORT adds |
| --- | --- | --- |
| [POGS](https://arxiv.org/pdf/2503.05189) (ICRA 2025) | Tracks objects by matching rendered DINO features of Gaussian clusters to images, one pose per cluster | Generic online MOT from RGB only; uncertainty rendered, not just pose |
| [GTR](https://arxiv.org/pdf/2505.11905) | 6-DoF tracking and reconstruction of one unknown object from RGB-D | Many objects, monocular, data association |
| [Paparazzo](https://www.themoonlight.io/review/paparazzo-active-mapping-of-moving-3d-objects) | EKF on SE(3) beside a per-object Gaussian model, RGB-D | Covariance folded into the splats; joint multi-track compositing |
| [Inverse-rendering 3D MOT](https://light.princeton.edu/wp-content/uploads/2025/08/inr_tracker.pdf) (Princeton) | Optimizes generative object latents through a renderer for 3D MOT | Explicit Gaussians, closed-form belief, no per-frame latent optimization |
| [PHALP](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2021/tracking_people_with_3d_representations__48ff8c0c.txt) (NeurIPS 2021) | 3D human mesh and texture for person tracking | Class-agnostic, Gaussian representation |
| [Inst4DGS](https://arxiv.org/abs/2603.18402) (2026) | Identity-consistent 4D Gaussians via Sinkhorn label permutation | Online and causal, not per-scene optimization |
| [BA-GS](https://cvpr.thecvf.com/virtual/2026/poster/39464) (CVPR 2026) | Kalman filter on Gaussian positions during reconstruction | Filter on object pose, used for association |
| DepthSplat, Depth Anything 3 | Depth priors inside feed-forward Gaussian prediction | Depth used for lifting and occlusion order in tracking |

The claim rests on targeted searches, not an exhaustive review. A dedicated arXiv sweep for 2025 to 2026 is a Phase 0 task.

## System architecture

A frame flows from perception on the left into association in the middle; track state on the right persists from frame to frame.

&#91;embedded content: SplatSORT architecture · one frame, state loop on the right\]

Ego-motion re-expresses track states in the current camera before the belief transform. Unmatched detections become tentative tracks through the object Gaussian encoder, inside the lifecycle step.

## Core method: belief splats

Pose uncertainty is folded into each Gaussian in closed form, so one render pass yields where every track is expected to be visible and how confident that expectation is.

**Track representation.** Track k holds object-frame Gaussians {μᵢ, Σᵢ, αᵢ, cᵢ, fᵢ}: mean, covariance, opacity, color and a distilled feature. Its filter state gives a pose estimate (R̂, t̂) with translation covariance Pₜ and small-angle rotation covariance P\_θ.

**Belief transform.** Translation uncertainty convolves exactly with each Gaussian. Rotation uncertainty is linearized about the object center.

```latex
\mu_i^{b} = \hat R\,\mu_i + \hat t
```

```latex
\Sigma_i^{b} = \hat R\,\Sigma_i\,\hat R^{\top} + P_t + J_i P_\theta J_i^{\top}, \qquad J_i = -[\hat R\,\mu_i]_{\times}
```

```latex
\alpha_i^{b} = \alpha_i \sqrt{\det(\hat R\,\Sigma_i\,\hat R^{\top}) \,/\, \det(\Sigma_i^{b})}
```

The opacity correction keeps each primitive's total mass constant as it spreads. The result is exact for translation at the density level, first-order for rotation, and approximate after alpha compositing, which is nonlinear.

**Joint render.** All tracks' belief splats are rasterized together in depth order, with a track-ID channel. Per pixel u this gives each track's visible occupancy Oₖ(u), its rendered feature Fₖ(u), and its unoccluded occupancy Õₖ(u) from a solo render.

**Visibility.** The visible fraction Vₖ = ΣOₖ / ΣÕₖ says how much of a track should be seen. A track with low Vₖ and no matching detection is explained as occluded, not missed.

**Association cost.** For track k and detection j (mask Mⱼ or box), with weights λ learned on training data:

```latex
C_{kj} = -\big(\lambda_1 \ell_{\text{occ}} + \lambda_2 \ell_{\text{app}} + \lambda_3 \ell_{\text{depth}} + \lambda_4 \ell_{\text{geo}}\big)
```

- **ℓ\_occ:** soft Dice between Oₖ and Mⱼ. It is already occlusion-aware because Oₖ is the visible part.
- **ℓ\_app:** cosine similarity between visibility-weighted Fₖ and the detection's features inside Mⱼ.
- **ℓ\_depth:** Mahalanobis distance between the detection's lifted 3D position and the track's predicted position under Pₜ.
- **ℓ\_geo:** normalized LightGlue inlier count between the track rendered at its predicted pose and the detection crop.

**Why it helps.** As a track goes unobserved, Pₜ grows and its belief splat widens and dims. Matching becomes spatially permissive but less confident, which replaces DeepSORT's hand-tuned matching cascade by track age.

## Component specifications

Every component is off-the-shelf except the object Gaussian encoder head, the belief renderer and the learned association weights.

| Component | Default choice | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| Detector | YOLOX, as in ByteTrack and DanceTrack baselines | Frame | Boxes, scores | Kept identical to baselines for fair comparison |
| Mask head | SAM2 prompted by boxes | Frame, boxes | Instance masks | Optional; box ellipses as fallback |
| Depth | Depth Anything V2 metric or Depth Anything 3 | Frame | Metric depth map | Per-frame scale aligned over time on static background |
| Camera motion | Background LightGlue matches, essential matrix | Frames t-1, t | Ego-motion | DA3 pose output as an alternative |
| Object Gaussian encoder | Single-view, pixel-aligned Gaussians from crop plus depth | Crop, mask, depth | N\_g Gaussians with DINOv2 features | N\_g is a tuning parameter; one Gaussian per track is the minimal mode |
| Track state | 3D Kalman filter, constant velocity | Lifted measurements | Pose, velocity, Pₜ, P\_θ | State: position, velocity, yaw, yaw rate, log scale |
| Belief renderer | gsplat-style CUDA rasterizer with a track-ID channel | Belief splats, camera | Oₖ, Õₖ, Fₖ, RGB per track | Rendered only inside gated windows |
| Geometric verifier | ALIKED or SuperPoint features with LightGlue | Rendered track view, detection crop | Inlier count | Run only when the cost margin between top candidates is small |
| Assignment | Hungarian with chi-square gating, two-stage high and low score | Cost matrix | Matches | ByteTrack-style second pass for low-score detections |
| Track update | Kalman update, then Gaussian refinement | Matched pair | Updated track | Refinement: a few gradient steps on mask and photometric loss, or confidence-weighted fusion of fresh crop Gaussians |
| Lifecycle manager | State machine | Matches, Vₖ | Track states | See below |

**Track lifecycle**

1. **Tentative:** created from an unmatched high-score detection; confirmed after a few consecutive matches.
2. **Confirmed:** matched normally; Gaussians refined on each match.
3. **Occluded:** unmatched but Vₖ below threshold; predicted forward, not aged.
4. **Lost:** unmatched while expected visible; kept in a gallery for re-identification by rendering at a new detection's pose.
5. **Deleted:** lost for longer than the gallery horizon.

## Per-frame algorithm

Each frame runs perceive, predict, render, associate, update and manage, strictly causally.

1. **Perceive.** Run the detector, mask head and depth model on frame t.
2. **Compensate.** Estimate ego-motion from background matches; express track states in the current camera frame.
3. **Lift.** Back-project each detection's mask through the depth map to a 3D centroid, extent and covariance.
4. **Predict.** Propagate each track's Kalman state; Pₜ and P\_θ grow with elapsed time.
5. **Build belief splats.** Apply the belief transform to every active track's Gaussians.
6. **Render.** Rasterize all belief splats jointly in depth order; record Oₖ, Fₖ, and Õₖ from solo renders; compute Vₖ.
7. **Gate.** Discard pairs failing the chi-square test on 3D position.
8. **Score.** Compute ℓ\_occ, ℓ\_app and ℓ\_depth for gated pairs; add ℓ\_geo from LightGlue only for ambiguous pairs.
9. **Assign.** Hungarian on high-score detections, then a second pass on low-score detections against remaining tracks.
10. **Update.** Kalman update on matched pairs; refine or fuse their Gaussians.
11. **Manage.** Move unmatched tracks to occluded or lost based on Vₖ; spawn tentative tracks from unmatched high-score detections via the encoder; try gallery re-identification before spawning.
12. **Emit.** Output boxes and IDs in the image plane, plus optional 3D poses.

## Training strategy and data

The system is mostly training-free; three small learned parts are trained on standard tracking and multi-view data, with no 3D labels required.

| Learned part | Objective | Data | Supervision |
| --- | --- | --- | --- |
| Object Gaussian encoder | Render loss on held-out views of the same object | Multi-view object datasets such as CO3D or Objaverse renders; fine-tuned on tracking crops | Photometric and mask loss; DINOv2 feature distillation |
| Association weights λ and calibration | Logistic loss on match vs non-match pairs | MOT17, DanceTrack, SportsMOT train splits | Ground-truth identities |
| Visibility threshold and gallery horizon | Grid search on HOTA | Same train splits | Ground-truth identities |

**Self-supervised extension.** Tracklets from a baseline tracker give pseudo-identities on unlabeled video. Cross-frame render consistency of the same track then trains the encoder without manual labels.

**Frozen components.** Detector, mask head, depth model, keypoint extractor and LightGlue are used as released.

## Evaluation plan

DanceTrack is the primary benchmark because it isolates the failure SplatSORT targets: look-alike objects, nonlinear motion and frequent occlusion.

**Benchmarks**

| Benchmark | Why it matters |
| --- | --- |
| DanceTrack | Uniform appearance, heavy occlusion; 2D ReID is weakest here |
| SportsMOT | Fast motion, similar uniforms |
| MOT17 and MOT20 | Standard pedestrian tracking; MOT20 stresses small, dense objects |
| BDD100K MOT | Driving, multiple classes, strong camera motion |

**Metrics.** HOTA with its AssA and DetA parts, IDF1 and ID switches. On MOT17 and MOT20, which annotate visibility, each is also reported on an occlusion slice: boxes with visibility below 50%.

**Baselines.** ByteTrack, OC-SORT, BoT-SORT and Deep OC-SORT, all run on identical detections.

**Ablations**

| Variant | Question answered |
| --- | --- |
| Point-estimate render (no belief transform) | Does rendering uncertainty matter? |
| Independent per-track renders (no joint compositing) | Does occlusion reasoning matter? |
| No LightGlue term | Does geometric verification add over features? |
| 2D Kalman instead of 3D | Does depth lifting matter? |
| No depth model (constant depth prior) | How sensitive is the system to depth quality? |
| N\_g = 1, 64, 256, 1024 | How many Gaussians does association need? |
| No visibility-aware lifecycle | Does explaining occlusion reduce ID switches? |

## Compute and latency budget

Added cost comes from per-frame depth inference and from work that scales with the number of tracks; no latency figures exist yet, so Phase 1 profiling sets the targets.

| Stage | Scales with | Mitigation |
| --- | --- | --- |
| Detector and mask head | Frame resolution | Unchanged from baselines |
| Depth model | Frame resolution | Smaller backbone; run at reduced resolution; reuse on static frames |
| Object Gaussian encoder | New tracks per frame | Runs only on spawn and periodic refresh |
| Belief render | Tracks × N\_g × window area | Render only gated windows; small N\_g |
| LightGlue verification | Ambiguous pairs | Skip when the cost margin is large; early exit and point pruning built into LightGlue |
| Gaussian refinement | Matched tracks × steps | Few steps, or fusion instead of gradient steps |
| Hungarian assignment | Tracks × detections | Negligible after gating |

## Risks and mitigations

The largest risk is weak Gaussians on small or non-rigid objects; the design degrades gracefully to a single Gaussian per track, which is still a 3D, occlusion-aware DeepSORT.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Small objects (tens of pixels tall) give poor crop Gaussians | Weak ℓ\_app and ℓ\_geo | Fall back to N\_g = 1 ellipsoid; rely on ℓ\_occ and ℓ\_depth |
| Non-rigid objects (dancers, players) break the rigid-body assumption | Render misaligns with the detection | Frequent appearance refresh; inflate Σᵢ; coarse part groups as a later option |
| Monocular depth scale drifts between frames | Wrong 3D motion and gating | Temporal scale alignment on static background; use relative depth only for compositing order |
| Alpha compositing makes the belief approximate | Miscalibrated likelihoods | Learn a logistic calibration of C on training data |
| Compute grows with track count | Below real-time on crowded scenes | Gated windows, small N\_g, LightGlue only on ambiguous pairs |
| Concurrent work publishes the same idea | Novelty loss | Phase 0 literature sweep; position on the closed-form belief and joint compositing |
| Gains appear only on DanceTrack | Narrow contribution | Report all four benchmarks and the occlusion slice |

## Roadmap

The build is staged so the core hypothesis is tested with the smallest possible system before any heavy components are added; no dates are set yet.

&#91;embedded content: Roadmap · 5 phases, 4 gates\]

If Gate 1 fails, the belief transform does not help on its own, and later phases should be re-scoped before investing in them.

## Open questions

- What throughput target makes the system useful: offline analytics, or real time at 25 to 30 FPS?
- Is a mask head required, or do box-derived ellipses give enough signal for ℓ\_occ?
- How should non-rigid objects be modeled beyond appearance refresh: part groups, or a deformation field per track?
- Should the background be rendered as an occluder, or is track-to-track occlusion enough?
- Which N\_g gives the best association per unit of compute on small objects?
- Is single-view Gaussian encoding sufficient, or should each track accumulate views over time from the start?

## References

- [POGS: Persistent Object Gaussian Splat](https://arxiv.org/pdf/2503.05189), ICRA 2025
- [GTR: Gaussian Splatting Tracking and Reconstruction of Unknown Objects](https://arxiv.org/pdf/2505.11905)
- [Paparazzo: Active Mapping of Moving 3D Objects](https://www.themoonlight.io/review/paparazzo-active-mapping-of-moving-3d-objects) (review page)
- [Inverse-rendering 3D multi-object tracking](https://light.princeton.edu/wp-content/uploads/2025/08/inr_tracker.pdf), Princeton Computational Imaging Lab
- [Tracking People with 3D Representations (PHALP)](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2021/tracking_people_with_3d_representations__48ff8c0c.txt), NeurIPS 2021
- [Inst4DGS: Instance-Decomposed 4D Gaussian Splatting](https://arxiv.org/abs/2603.18402), 2026
- [BA-GS: Bayesian Adaptive Gaussian Splatting](https://cvpr.thecvf.com/virtual/2026/poster/39464), CVPR 2026
- [EmbodiedSplat: Online Feed-Forward Semantic 3DGS](https://arxiv.org/pdf/2603.04254), 2026
- [SLARM: Streaming and Language-Aligned Reconstruction Model](https://arxiv.org/pdf/2603.22893), 2026
- [Dynamic 3D Representations for Visual Tracking](https://publications.rwth-aachen.de/record/998170/files/998170.pdf), Luiten, RWTH Aachen thesis
- Also cited by name: 3D Gaussian Splatting (Kerbl et al., 2023), DeepSORT, ByteTrack, OC-SORT, BoT-SORT, LightGlue, Depth Anything V2 and 3, DepthSplat, AnySplat, SAM2, DINOv2.