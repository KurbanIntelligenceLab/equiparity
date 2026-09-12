import Mathlib.LinearAlgebra.Dimension.Finrank
import Mathlib.GroupTheory.GroupAction.Basic
import Mathlib.Algebra.Ring.Parity
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-! Abstract symmetry guarantees for `The parity gap in crystal tensor prediction`.
The representation and input-action hypotheses are explicit. No neural-network
implementation, floating-point computation or crystallographic classification is assumed verified.
-/

namespace ParityGap

section Periodic
variable {ι P S W : Type*} [AddCommGroup P]

/-- Inversion about an arbitrary centre agrees with a species-preserving relabelling
and individual lattice translations. A1 and both parts of A2 imply the input identity. -/
theorem periodic_inversion_identity
    (species : ι → S) (L : AddSubgroup P) (p : ι → P) (f : (ι → P) → W)
    (permInv : ∀ q (π : Equiv.Perm ι), (∀ i, species (π i) = species i) →
      f (fun i => q (π i)) = f q)
    (latticeInv : ∀ (q t : ι → P), (∀ i, t i ∈ L) → f (fun i => q i + t i) = f q)
    (rigidInv : ∀ q c, f (fun i => q i + c) = f q)
    (c : P) (π : Equiv.Perm ι) (t : ι → P)
    (speciesPreserved : ∀ i, species (π i) = species i)
    (latticeShifts : ∀ i, t i ∈ L)
    (centrosymmetric : ∀ i, (2 : ℕ) • c - p i = p (π i) + t i) :
    f (fun i => -p i) = f p := by
  have hc : (fun i => -p i) = (fun i => (p (π i) + t i) + -((2 : ℕ) • c)) := by
    funext i
    rw [← centrosymmetric i]
    abel
  rw [hc, rigidInv, latticeInv _ _ latticeShifts, permInv _ _ speciesPreserved]
end Periodic

section Linear
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Over the reals, a vector equal to its negative is zero. -/
theorem eq_neg_forces_zero {v : V} (h : v = -v) : v = 0 := by
  have hz : (2 : ℝ) • v = 0 := by
    calc
      (2 : ℝ) • v = v + v := two_smul ℝ v
      _ = -v + v := congrArg (fun w => w + v) h
      _ = 0 := neg_add_cancel v
  exact (smul_eq_zero.mp hz).resolve_left (by norm_num)

/-- Invariance of a centrosymmetric input and odd output parity force zero. -/
theorem symmetry_forced_zero {X : Type*} (inv : X → X) (f : X → V) (x : X)
    (inputIdentity : f (inv x) = f x) (oddParity : f (inv x) = -f x) :
    f x = 0 := eq_neg_forces_zero (inputIdentity.symm.trans oddParity)

/-- Cartesian rank-r inversion multiplies each of the r tensor indices by -1. -/
def polarInversion (r : ℕ) (T : (Fin r → Fin 3) → ℝ) : (Fin r → Fin 3) → ℝ :=
  fun a => (∏ _ : Fin r, (-1 : ℝ)) * T a

theorem polarInversion_sign (r : ℕ) (T : (Fin r → Fin 3) → ℝ) :
    polarInversion r T = (-1 : ℝ) ^ r • T := by
  funext a
  simp [polarInversion]

variable {G : Type*} [Group G]

/-- Invariant vectors of the restricted representation, including any physical
index-symmetry subspace already chosen as the representation space V. -/
def fixedSpace (D : G →* Module.End ℝ V) (H : Subgroup G) : Submodule ℝ V where
  carrier := {v | ∀ g ∈ H, D g v = v}
  zero_mem' := by simp
  add_mem' := by
    intro a b ha hb g hg
    simp [map_add, ha g hg, hb g hg]
  smul_mem' := by
    intro a v hv g hg
    simp [hv g hg]

/-- If inversion acts trivially and every full-group element is proper or inversion
 times proper, the full and proper invariant spaces coincide. -/
theorem fixedSpace_even (D : G →* Module.End ℝ V) (H : Subgroup G) (inv : G)
    (cover : ∀ g, g ∈ H ∨ ∃ h ∈ H, g = inv * h)
    (evenParity : ∀ v, D inv v = v) : fixedSpace D ⊤ = fixedSpace D H := by
  ext v
  constructor
  · intro hv g hg
    exact hv g (by trivial)
  · intro hv g _
    rcases cover g with hg | ⟨h, hh, rfl⟩
    · exact hv g hg
    · rw [map_mul]
      change D inv (D h v) = v
      rw [hv h hh, evenParity]

/-- Odd inversion kills the full invariant space. -/
theorem fixedSpace_odd (D : G →* Module.End ℝ V) (inv : G)
    (oddParity : ∀ v, D inv v = -v) : fixedSpace D ⊤ = ⊥ := by
  ext v
  constructor
  · intro hv
    have h := hv inv (by trivial)
    exact eq_neg_forces_zero (h.symm.trans (oddParity v))
  · intro hv
    have hz : v = 0 := hv
    subst v
    simp [fixedSpace]

/-- Dimension difference used in the manuscript, for finite-dimensional real V. -/
noncomputable def gap [FiniteDimensional ℝ V]
    (D : G →* Module.End ℝ V) (H : Subgroup G) : ℕ :=
  Module.finrank ℝ (fixedSpace D H) - Module.finrank ℝ (fixedSpace D ⊤)

theorem gap_even [FiniteDimensional ℝ V]
    (D : G →* Module.End ℝ V) (H : Subgroup G) (inv : G)
    (cover : ∀ g, g ∈ H ∨ ∃ h ∈ H, g = inv * h)
    (evenParity : ∀ v, D inv v = v) : gap D H = 0 := by
  unfold gap
  rw [fixedSpace_even D H inv cover evenParity, Nat.sub_self]

theorem gap_odd [FiniteDimensional ℝ V]
    (D : G →* Module.End ℝ V) (H : Subgroup G) (inv : G)
    (oddParity : ∀ v, D inv v = -v) :
    gap D H = Module.finrank ℝ (fixedSpace D H) := by
  unfold gap
  rw [fixedSpace_odd D inv oddParity]
  simp

/-- The full invariant space is contained in the proper invariant space. -/
theorem fixedSpace_full_le (D : G →* Module.End ℝ V) (H : Subgroup G) :
    fixedSpace D ⊤ ≤ fixedSpace D H := by
  intro v hv g _
  exact hv g (by trivial)

/-- Even-rank polar tensors have zero additional inversion constraint. -/
theorem parity_gap_even_rank [FiniteDimensional ℝ V]
    (D : G →* Module.End ℝ V) (H : Subgroup G) (inv : G) (r : ℕ)
    (cover : ∀ g, g ∈ H ∨ ∃ h ∈ H, g = inv * h)
    (polarParity : ∀ v, D inv v = (-1 : ℝ) ^ r • v) (rankEven : Even r) :
    gap D H = 0 := by
  apply gap_even D H inv cover
  intro v
  rw [polarParity, rankEven.neg_one_pow, one_smul]

/-- At odd rank, the full invariant space vanishes and the gap is the entire
proper invariant dimension. -/
theorem parity_gap_odd_rank [FiniteDimensional ℝ V]
    (D : G →* Module.End ℝ V) (H : Subgroup G) (inv : G) (r : ℕ)
    (polarParity : ∀ v, D inv v = (-1 : ℝ) ^ r • v) (rankOdd : Odd r) :
    fixedSpace D ⊤ = ⊥ ∧ gap D H = Module.finrank ℝ (fixedSpace D H) := by
  have oddParity : ∀ v, D inv v = -v := by
    intro v
    rw [polarParity, rankOdd.neg_one_pow, neg_one_smul]
  exact ⟨fixedSpace_odd D inv oddParity, gap_odd D H inv oddParity⟩

section Orbit
variable {X : Type*} [MulAction G X]

/-- Equivariance constrains a value to the fixed space of the input stabilizer. -/
theorem value_in_fixedSpace (D : G →* Module.End ℝ V) (f : X → V)
    (equivariant : ∀ g x, f (g • x) = D g (f x)) (x : X) :
    f x ∈ fixedSpace D (MulAction.stabilizer G x) := by
  intro g hg
  have h : g • x = x := hg
  rw [← equivariant, h]

/-- The orbit construction is independent of the chosen group representative. -/
theorem orbit_value_well_defined (D : G →* Module.End ℝ V) (x : X) (v : V)
    (fixed : v ∈ fixedSpace D (MulAction.stabilizer G x))
    (g h : G) (same : g • x = h • x) : D g v = D h v := by
  have hs : h⁻¹ * g ∈ MulAction.stabilizer G x := by
    change (h⁻¹ * g) • x = x
    rw [mul_smul, same, inv_smul_smul]
  calc
    D g v = D (h * (h⁻¹ * g)) v := by simp
    _ = D h (D (h⁻¹ * g) v) := by rw [map_mul]; rfl
    _ = D h v := by rw [fixed _ hs]

/-- Extend an invariant vector along its orbit, and use zero off that orbit. -/
noncomputable def orbitExtension (D : G →* Module.End ℝ V) (x : X) (v : V) (y : X) : V := by
  classical
  exact if h : ∃ g : G, g • x = y then D h.choose v else 0

theorem orbitExtension_on_orbit (D : G →* Module.End ℝ V) (x : X) (v : V)
    (fixed : v ∈ fixedSpace D (MulAction.stabilizer G x)) (g : G) :
    orbitExtension D x v (g • x) = D g v := by
  classical
  have h : ∃ h : G, h • x = g • x := ⟨g, rfl⟩
  rw [orbitExtension, dif_pos h]
  exact orbit_value_well_defined D x v fixed _ g h.choose_spec

/-- Every stabilizer-invariant vector is attained by an equivariant function.
There is deliberately no continuity or neural-network-realizability conclusion. -/
theorem invariant_value_attainable (D : G →* Module.End ℝ V) (x : X) (v : V)
    (fixed : v ∈ fixedSpace D (MulAction.stabilizer G x)) :
    ∃ f : X → V, f x = v ∧ ∀ g y, f (g • y) = D g (f y) := by
  classical
  refine ⟨orbitExtension D x v, ?_, ?_⟩
  · simpa using orbitExtension_on_orbit D x v fixed (1 : G)
  · intro g y
    by_cases hy : ∃ h : G, h • x = y
    · obtain ⟨h, rfl⟩ := hy
      rw [← mul_smul, orbitExtension_on_orbit D x v fixed,
        orbitExtension_on_orbit D x v fixed, map_mul]
      rfl
    · have hgy : ¬ ∃ h : G, h • x = g • y := by
        rintro ⟨h, hh⟩
        apply hy
        refine ⟨g⁻¹ * h, ?_⟩
        rw [mul_smul, hh, inv_smul_smul]
      simp [orbitExtension, hy, hgy]
end Orbit

/-- Explicit odd output enforcement, independent of internal feature parity. -/
noncomputable def antisymmetrize {X : Type*} (inv : X → X) (f : X → V) (x : X) : V :=
  (2 : ℝ)⁻¹ • (f x - f (inv x))

theorem antisymmetrize_zero {X : Type*} (inv : X → X) (f : X → V) (x : X)
    (inputIdentity : f (inv x) = f x) : antisymmetrize inv f x = 0 := by
  simp [antisymmetrize, inputIdentity]

theorem antisymmetrize_odd {X : Type*} (inv : X → X) (f : X → V)
    (involutive : Function.Involutive inv) (x : X) :
    antisymmetrize inv f (inv x) = -antisymmetrize inv f x := by
  simp only [antisymmetrize, involutive x]
  rw [← smul_neg, neg_sub]
end Linear
end ParityGap
