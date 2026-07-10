# F1 — related work and framing

Every characterisation below was checked against the cited paper's abstract before being written.
Where a paper does something adjacent but not identical to what we do, the difference is stated
rather than blurred.

## The symmetry-breaking thread: they relax equivariance, we test whether it holds

There is an active line of work on letting equivariant networks produce outputs *less* symmetric
than their inputs.

- **Smidt, Geiger & Miller, "Finding Symmetry Breaking Order Parameters with Euclidean Neural
  Networks"** ([arXiv:2007.02005](https://arxiv.org/abs/2007.02005)). Opens with Curie's principle —
  "when effects show certain asymmetry, this asymmetry must be found in the causes that gave rise to
  them" — and shows that an equivariant network cannot preferentially fit an output that is not
  symmetrically compatible with its input; it weighs symmetrically degenerate possibilities equally.
  They exploit this by *learning the symmetry-breaking input* (deforming a square into a rectangle;
  generating octahedral tilting in perovskites).

  **Relation to us.** This paper establishes that equivariant networks uphold Curie's principle. It
  does not address parity, inversion, or improper operations (we checked: they appear nowhere in the
  abstract). Our contribution is on the physics side of the same coin: for *tensor properties*,
  upholding Curie's principle **requires** equivariance to improper operations, and SO(3) models —
  which the field routinely calls "E(3)-equivariant" — do not have it. We show the principle they
  rely on is silently violated by a widely deployed subclass.

- **Xie & Smidt, "Equivariant Symmetry Breaking Sets"**
  ([arXiv:2402.02681](https://arxiv.org/abs/2402.02681)). "By construction ENNs cannot produce lower
  symmetry outputs given a higher symmetry input." They inject *symmetry-breaking objects* whose own
  equivariance is constrained, and relate minimising the size of these sets to a group-theory problem
  solved for point groups.

- **Kaba & Ravanbakhsh, "Symmetry Breaking and Equivariant Neural Networks"**
  ([arXiv:2312.09016](https://arxiv.org/abs/2312.09016)). Introduces a *relaxation* of equivariance
  to permit symmetry breaking, and builds equivariant MLPs that can break symmetry.

- **"Improving Equivariant Networks with Probabilistic Symmetry Breaking"**
  ([arXiv:2503.21985](https://arxiv.org/abs/2503.21985)), 2025. Same thread, probabilistic treatment.

**The complement.** That literature asks how to *escape* the constraint when a physical system
spontaneously breaks symmetry. We ask what happens when a model escapes the constraint it was
supposed to keep. For a parity-odd tensor on a centrosymmetric crystal there is no spontaneous
symmetry breaking to model — the answer is exactly zero — and a nonzero prediction is not a feature
but an impossibility. Our E2 experiment sits precisely at the join: sweep a crystal continuously from
centrosymmetric to polar, and the O(3) model's response switches on exactly when the symmetry
switches off, while the SO(3) model's spurious floor was there the whole time.

## Readout-level fixes exist, and they are not what we are diagnosing

Two recent crystal-tensor methods enforce the right symmetry at the *output*:

- **GMTNet** — "A Space Group Symmetry Informed Network for O(3) Equivariant Crystal Tensor
  Prediction" ([arXiv:2406.12888](https://arxiv.org/abs/2406.12888), ICML 2024). Predicts dielectric,
  piezoelectric and elastic tensors; its four modules end in an explicit **symmetry enforcement
  module**, so predictions are "fully consistent with the intrinsic crystal symmetries."

- **GoeCTP** — "Fast Crystal Tensor Property Prediction: A General O(3)-Equivariant Framework Based
  on Polar Decomposition" ([arXiv:2410.02372](https://arxiv.org/abs/2410.02372)). An external
  **rotation-and-reflection module** based on polar decomposition canonicalises the crystal into a
  standardised position; it is plug-and-play on top of any scalar property network and imposes no
  equivariance constraint on the architecture.

**One sentence of distinction.** These are readout-level and canonicalisation-level remedies: they
repair the output of a model whose *features* remain parity-blind. Our diagnosis is at the feature
level, and it is what determines whether the zero is structural — true for any weights, at random
initialisation, on any crystal — or merely enforced after the fact for the one quantity someone
remembered to enforce it on. E4 makes the cost concrete: symmetrising an SO(3) model's output does
remove its false flags, but only because inversion maps a centrosymmetric crystal to itself, and it
leaves every other quantity derived from those features unrepaired. We introduce **no new baselines**
against GMTNet or GoeCTP; the comparison here is conceptual, not empirical.

## Why this matters now: tensor readouts on universal-potential embeddings

The pattern of attaching a property head to a pretrained universal interatomic potential is
established — e.g. dielectric-tensor prediction from the latent embeddings of a universal neural
network potential with an equivariant readout decoder
([npj Comput. Mater. 10, 2024](https://www.nature.com/articles/s41524-024-01450-z)), and pipelines
that extract fixed-length embeddings from NIP foundation models for downstream property prediction.

**State the limit of this argument honestly.** The dielectric tensor is rank-2 and parity-**even**,
so the failure mode we document does not arise for it, and that paper is not an example of the bug.
It is evidence that the *pattern* is now standard practice. The hazard is what happens when a
parity-**odd** readout (piezoelectric, and any odd-rank response) is attached to a backbone whose
features carry no parity label. Table 1 (`docs/reports/checkpoint8_prevalence_audit.md`) shows that
the SO(3)-only group includes EquiformerV2 and the eSCN family — currently among the most widely
deployed backbones — and that the distinction is nowhere in their configuration surface:
EquiformerV2 exposes `lmax`, not a parity flag. NequIP is the sharpest case, since it *has* a
`parity` boolean whose own docstring shows it does not strip parity labels.

The backbone's symmetry group silently decides whether physically impossible predictions are
possible. That is the sentence this paper exists to justify.
