# Fairness Audit with Grad-CAM

## Why Grad-CAM

Grad-CAM is used in this project as a model auditing tool.

In many computer vision workflows, Grad-CAM is used only to produce visually appealing explanations of model decisions. Here, it serves a more critical purpose: to inspect whether the model is focusing on semantically meaningful regions when classifying distracted driving behaviors and to examine whether that focus changes across different visual subgroups.

The key question is not simply “what did the model predict?” but also:

- where did the model attend?
- was that attention behaviorally relevant?
- did the attention pattern remain consistent across subgroup comparisons?

## Audit Setup

To support subgroup-based inspection, a smaller manually selected subset was created for each driving behavior class. For each behavior category, two small sets of images were assembled:

- White drivers
- Non-White drivers

These subsets were selected with attention to image quality and visual consistency in order to reduce obvious confounding effects when comparing heatmaps.

This setup does **not** constitute a demographic benchmark dataset, nor does it allow strong causal claims about bias. Instead, it provides a qualitative audit setting for inspecting potential attention differences.

## What the Audit Examines

The fairness-oriented Grad-CAM analysis focuses on whether the model attends to regions that are semantically relevant for distracted driving classification, such as:

- hands
- phone position
- face / mouth area
- steering wheel interaction
- driving-related objects

Potentially concerning attention behavior includes:

- excessive focus on background
- attention on clothing rather than action
- inconsistent focus across similar driving behaviors
- systematic subgroup differences in attention location

## Observed Pattern

The Grad-CAM comparisons suggest that attention behavior is not always uniform across subgroup examples.

In some examples, the model appears to focus on action-relevant regions such as hands or mouth areas. In others, especially within some Non-White subgroup examples, the heatmaps appear to shift toward less clearly relevant image regions.

This does **not** prove demographic bias by itself. However, it raises an important Responsible AI concern: the model may be relying on different visual shortcuts or unstable cues across subgroup contexts.

## Why This Matters

If a distracted driving model uses inconsistent or spurious attention strategies, that can create reliability and fairness risks in deployment.

Even if overall accuracy is strong, the system may still be problematic if:

- it relies on subgroup-specific artifacts
- it under-attends to the actual action in some settings
- it generalizes poorly across visual conditions

This is why interpretability should not be treated as optional. It provides evidence about model reasoning that standard performance metrics alone cannot capture.

## Interpretation Caution

This audit is intentionally qualitative and exploratory. The subgroup comparisons are useful for surfacing possible issues, but they do not establish definitive causal conclusions about demographic unfairness.

The appropriate interpretation is:

- Grad-CAM reveals attention behavior worth inspecting
- some patterns are consistent with possible shortcut learning or attention bias
- further quantitative fairness work would be needed to validate those concerns

## Takeaway

The main contribution of this audit is to show how a standard distracted driving classifier can be evaluated through a Responsible AI lens.

Rather than asking only whether the model is correct, the project asks whether the model appears to be correct **for the right reasons**.
