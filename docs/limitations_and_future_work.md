# Limitations and Future Work

## Limitations

### 1. Small subgroup audit sample

The fairness-oriented comparison relies on relatively small manually selected White and Non-White subsets. This makes the audit useful for qualitative interpretation, but not strong enough for statistically robust fairness claims.

### 2. No demographic ground-truth labels

The subgroup audit is based on visually inferred subgroup assignment rather than official demographic annotations. This limits the strength of any fairness conclusions and means the audit should be interpreted carefully.

### 3. Grad-CAM is coarse

Grad-CAM is useful for interpretability, but it is still a relatively coarse visualization method. It can suggest where the model is attending, but it does not provide a complete explanation of model reasoning.

### 4. ImageNet pretraining may carry upstream bias

Because the classifier is built on a pretrained VGG16 backbone, some problematic attention behavior may originate not only from the fine-tuning dataset, but also from biases inherited from large-scale pretraining data.

### 5. Qualitative rather than quantitative fairness analysis

This project is best understood as an interpretability-based fairness audit, not a complete fairness benchmarking pipeline. It surfaces concerns, but does not provide a full quantitative bias assessment across well-defined demographic groups.

### 6. Dataset realism vs. deployment realism

Although the State Farm dataset is widely used and useful for model development, it does not perfectly represent real deployment conditions. Lighting, camera angle, driver appearance, and environmental variation may differ substantially in production settings.

## Future Work

### 1. Expand subgroup evaluation

A stronger next step would be to build a larger and more systematically annotated subgroup evaluation set so that attention differences and performance differences could be analyzed more rigorously.

### 2. Add quantitative fairness metrics

Future work could complement Grad-CAM with subgroup-level performance comparisons such as:

- accuracy gaps
- false positive / false negative disparities
- calibration differences
- robustness across visual conditions

### 3. Compare multiple backbones

The current project uses VGG16, but future work could examine whether the same audit findings persist across other architectures such as ResNet, EfficientNet, or vision transformers.

### 4. Improve interpretability methods

Grad-CAM is a strong starting point, but interpretability could be extended with additional methods such as:

- occlusion sensitivity
- attention rollout
- feature attribution comparisons
- counterfactual visual perturbations

### 5. Investigate debiasing strategies

If attention instability or subgroup inconsistencies are confirmed, future work could explore mitigation approaches such as:

- balanced subgroup sampling
- augmentation strategies
- fairness-aware fine-tuning
- attention regularization

### 6. Connect auditing to deployment policy

A longer-term Responsible AI direction would be to connect visual auditing results to deployment decisions. For example, model auditing outcomes could help determine whether a system is suitable for deployment, requires subgroup-specific review, or should only be used with human oversight.

## Final Reflection

The main limitation of this project is also part of its value: it does not claim to solve fairness in distracted driving detection. Instead, it demonstrates a practical way to begin auditing model reasoning in a safety-critical computer vision task.

That makes the project a useful foundation for more rigorous Responsible AI evaluation in future work.
