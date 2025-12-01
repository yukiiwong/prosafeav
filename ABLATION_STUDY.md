# ProSafeAV Ablation Study: RSSM Variants

## Overview

This document describes the ablation study design for understanding the role of **stochastic latent variables** in world models for ProSafeAV.

## Research Questions

1. **Is a lightweight RSSM sufficient?**
   - Can we reduce model complexity while maintaining performance?
   - What is the trade-off between model size and effectiveness?

2. **Is stochasticity necessary in latent space?**
   - Does removing stochastic sampling hurt performance?
   - Given that EVT handles uncertainty, is latent stochasticity redundant?

3. **How do these variants compare to full DreamerV3?**
   - Performance vs. computational cost trade-off
   - Sample efficiency comparison

---

## Variant Descriptions

### **Baseline: DreamerV3 (Full RSSM)**

**Architecture**:
```
RSSM: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
      z_t ~ p(z_t | h_t, e_t)
```

**Characteristics**:
- Stochastic dim: 32 (typical)
- Deterministic dim: 512+ (large)
- Full JAX implementation
- Complex, state-of-the-art

**Pros**:
- ✅ Best performance
- ✅ Handles uncertainty well
- ✅ State-of-the-art results

**Cons**:
- ❌ Complex to understand
- ❌ High computational cost
- ❌ Requires JAX expertise

---

### **Variant 1: ProSafeAV-RSSM (Lightweight)**

**Architecture**:
```
RSSM (simplified):
  h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

  Prior:     z_t ~ p(z_t | h_t)
  Posterior: z_t ~ q(z_t | h_t, e_t)

  Loss: reconstruction + KL(q||p) + reward
```

**Characteristics**:
- Stochastic dim: **16** (reduced from 32)
- Deterministic dim: **64** (reduced from 512+)
- Network sizes: 256→128→64 (reduced from 512+)
- PyTorch implementation
- Maintains RSSM structure

**Key Differences from DreamerV3**:
| Component | DreamerV3 | ProSafeAV-RSSM |
|-----------|-----------|----------------|
| Stochastic dim | 32 | **16** |
| Deterministic dim | 512+ | **64** |
| Hidden layers | 512+ | **128/64** |
| Framework | JAX | **PyTorch** |
| Complexity | Very High | **Medium** |

**Hypothesis**:
> "A lightweight RSSM is sufficient for CarDreamer tasks with low-dimensional observations (birdeye_wpt). The full complexity of DreamerV3 may be unnecessary."

**Expected Outcomes**:
- ✅ Faster training (smaller networks)
- ✅ Lower memory usage
- ✅ Easier to understand and debug
- ⚠️ Potentially slightly lower performance
- ⚠️ May need more training steps

---

### **Variant 2: ProSafeAV-Deterministic (No Stochastic)**

**Architecture**:
```
Deterministic dynamics:
  h_t = GRU(h_{t-1}, [e_t, a_{t-1}])

  NO sampling!
  NO z_t!

  Loss: reconstruction + consistency + reward
```

**Characteristics**:
- **NO** stochastic latent variables
- **NO** sampling or distributions
- **NO** KL divergence loss
- Only deterministic recurrent state (GRU)
- Hidden dim: 128

**Key Differences from RSSM**:
| Component | RSSM | ProSafeAV-Deterministic |
|-----------|------|-------------------------|
| Stochastic latent z | ✅ Yes | ❌ **No** |
| Deterministic latent h | ✅ Yes | ✅ Yes |
| Sampling | ✅ Yes | ❌ **No** |
| KL divergence | ✅ Yes | ❌ **No** |
| Uncertainty modeling | Latent space | **EVT module** |

**Hypothesis**:
> "Stochastic latent variables are NOT necessary when external uncertainty is handled by EVT. A purely deterministic latent dynamics model is sufficient for safe autonomous driving with ProSafeAV."

**Rationale**:
1. **EVT already handles uncertainty**: TTC and DRAC distributions capture safety-critical uncertainty
2. **Deterministic is simpler**: Easier to interpret, debug, and deploy
3. **Faster inference**: No sampling required
4. **Autonomous driving context**: Many driving scenarios are relatively deterministic

**Expected Outcomes**:
- ✅ Simplest model architecture
- ✅ Fastest inference (no sampling)
- ✅ Easiest to interpret
- ⚠️ May struggle with multi-modal behaviors
- ⚠️ Performance depends on EVT compensating for lack of latent uncertainty

---

## Comparison Matrix

| Aspect | DreamerV3 | ProSafeAV-RSSM | ProSafeAV-Deterministic |
|--------|-----------|----------------|------------------------|
| **Architecture** | | | |
| Stochastic latent | ✅ (32 dim) | ✅ (16 dim) | ❌ None |
| Deterministic latent | ✅ (512+ dim) | ✅ (64 dim) | ✅ (128 dim) |
| KL divergence | ✅ | ✅ | ❌ |
| Sampling | ✅ | ✅ | ❌ |
| **Complexity** | | | |
| Total parameters | ~10M+ | ~1M | ~500K |
| Hidden layer size | 512+ | 128/64 | 128 |
| Code complexity | Very High | Medium | Low |
| **Training** | | | |
| Framework | JAX | PyTorch | PyTorch |
| Memory usage | High | Medium | Low |
| Training speed | Slow | Medium | Fast |
| **Inference** | | | |
| Latency | Medium | Medium | **Fastest** |
| Determinism | Stochastic | Stochastic | **Deterministic** |
| Interpretability | Low | Medium | **High** |
| **Expected Performance** | | | |
| Sample efficiency | Best | Good | Medium |
| Final performance | Best | Good | TBD |
| Robustness | Best | Good | TBD |

---

## Experimental Protocol

### **Training Configuration**

All variants use identical training settings for fair comparison:

```yaml
Environment: carla_wpt (with ProSafeAV EVT module)
Observation: birdeye_wpt
Batch size: 16
Batch length: 64
Replay buffer: 1M
Imagination horizon: 10-15 steps
Learning rate (model): 1e-3
Learning rate (policy): 3e-4
```

### **Evaluation Metrics**

1. **Performance Metrics**:
   - Episode return
   - Success rate (reaching destination)
   - Collision rate
   - TTC distribution
   - DRAC values

2. **Efficiency Metrics**:
   - Sample efficiency (steps to threshold)
   - Training time per 1M steps
   - Memory usage (GPU/CPU)
   - Inference latency

3. **Safety Metrics** (ProSafeAV-specific):
   - EVT risk distribution
   - Joint TTC-DRAC risk
   - Safety constraint violations

4. **Model Quality Metrics**:
   - Reconstruction error
   - Reward prediction accuracy
   - Latent consistency
   - (For RSSM variants) KL divergence

---

## Expected Results and Analysis

### **Hypothesis 1: Lightweight RSSM Sufficiency**

**Prediction**:
```
Performance: DreamerV3 > ProSafeAV-RSSM ≈ 90-95% of DreamerV3
Efficiency: ProSafeAV-RSSM >> DreamerV3 (3-5x faster)
```

**If confirmed**:
- Use ProSafeAV-RSSM as default for CarDreamer
- Reserve DreamerV3 for complex scenarios only

**If rejected**:
- Small networks insufficient for world modeling
- Complexity of DreamerV3 is necessary

---

### **Hypothesis 2: Stochasticity Not Required**

**Prediction**:
```
With EVT: ProSafeAV-Deterministic ≈ ProSafeAV-RSSM
Without EVT: ProSafeAV-Deterministic << RSSM variants
```

**Key Test**: Train both with and without EVT module

**If confirmed**:
- EVT successfully handles all uncertainty
- Simplest model is best for deployment
- **Paradigm shift**: External safety module + deterministic dynamics

**If rejected**:
- Latent stochasticity captures important variations
- RSSM remains necessary even with EVT
- Stochasticity and EVT are complementary

---

## Running the Experiments

### **ProSafeAV-RSSM (Lightweight)**
```bash
python dreamerv3/train_prosafeav_rssm.py \
  --task carla_wpt \
  --dreamerv3.logdir logs/prosafeav_rssm
```

### **ProSafeAV-Deterministic**
```bash
python dreamerv3/train_prosafeav_deterministic.py \
  --task carla_wpt \
  --dreamerv3.logdir logs/prosafeav_deterministic
```

### **Baseline (DreamerV3)**
```bash
./train_prosafeav.sh 2000 0 \
  --task carla_wpt \
  --dreamerv3.logdir logs/dreamerv3_baseline
```

---

## Implications for ProSafeAV

### **If Deterministic Works Well**:

**Scientific Impact**:
- Challenge conventional wisdom about world models
- Show EVT can replace latent stochasticity
- Simpler is better for safety-critical systems

**Practical Impact**:
- Deploy simpler, faster models
- Easier verification and validation
- Lower computational requirements for vehicles

**Future Work**:
- Extend to other safety-critical domains
- Combine with other uncertainty quantification methods
- Investigate when stochasticity IS necessary

---

### **If RSSM Still Required**:

**Scientific Impact**:
- Confirm importance of latent stochasticity
- Show EVT and RSSM are complementary
- Lightweight RSSM as good middle ground

**Practical Impact**:
- Use ProSafeAV-RSSM as efficient alternative
- Reserve full DreamerV3 for most critical scenarios
- Balance performance and efficiency

**Future Work**:
- Optimize lightweight RSSM further
- Investigate hybrid approaches
- Study where each variant excels

---

## Publications and Reporting

### **Ablation Study Table (for paper)**

```latex
\begin{table}
\caption{Ablation study on RSSM components in ProSafeAV}
\begin{tabular}{lcccc}
\toprule
Variant & Stochastic & Params & Return & Safety \\
\midrule
DreamerV3 (full) & ✓ (32) & 10M+ & 100\% & High \\
ProSafeAV-RSSM & ✓ (16) & 1M & XX\% & High \\
ProSafeAV-Det & ✗ & 500K & XX\% & High \\
\bottomrule
\end{tabular}
\end{table}
```

### **Key Claims to Verify**

1. "Lightweight RSSM achieves XX% of full DreamerV3 performance with 10x fewer parameters"
2. "Deterministic latent dynamics are sufficient when combined with EVT uncertainty modeling"
3. "ProSafeAV-Deterministic reduces inference latency by XX% while maintaining safety"

---

## Conclusion

This ablation study provides critical insights into:
1. The **necessary complexity** of world models for autonomous driving
2. The **role of stochasticity** in latent space
3. The **interaction** between world models and external safety modules (EVT)

Results will guide the design of **efficient, safe, and interpretable** world models for ProSafeAV.

---

## Files

- `dreamerv3/prosafeav_rssm_agent.py` - Lightweight RSSM implementation
- `dreamerv3/prosafeav_deterministic_agent.py` - Deterministic variant
- `dreamerv3/train_prosafeav_rssm.py` - Training script for RSSM
- `dreamerv3/train_prosafeav_deterministic.py` - Training script for Deterministic
- `ABLATION_STUDY.md` - This document
