# Project Overview

## Motivation

Distracted driving remains a major public safety problem, and computer vision models are increasingly proposed as a way to classify driver behavior automatically. In many applied machine learning projects, model performance is emphasized primarily through accuracy metrics. However, in safety-critical settings, predictive performance alone is not sufficient. It is also important to understand **how** a model makes decisions and whether those decisions are based on semantically meaningful cues.

This project approaches distracted driving detection from a Responsible AI perspective. Instead of treating the classifier as a black box, it uses Grad-CAM to examine where the model focuses when making predictions and whether those attention patterns remain interpretable and potentially fair across different visual subgroups.

## Core Question

The central question of this project is:

**Can Grad-CAM provide actionable insight into how a distracted driving classifier makes decisions, and can it help surface potential attention bias across visually different driver subgroups?**

This shifts the emphasis of the project from pure classification performance to model auditing.

## Project Goal

The goal of this project is to build and audit a distracted driving image classifier that can be evaluated not only by accuracy, but also by interpretability and subgroup-level attention behavior.

More specifically, the project aims to:

- fine-tune a pretrained VGG16 model for 10-class distracted driving classification
- evaluate model performance with standard classification metrics
- generate Grad-CAM heatmaps to inspect model attention
- compare attention maps across manually constructed White and Non-White driver subsets
- assess whether the model focuses on behaviorally relevant cues or potentially spurious image regions

## Why This Project Matters

This project matters because distracted driving detection systems may eventually be deployed in real-world environments where incorrect or unfair model behavior has consequences. Even a high-performing model can be problematic if it relies on irrelevant visual shortcuts, fails to generalize across different users, or behaves inconsistently across subgroups.

By framing interpretability as an auditing task rather than a purely explanatory add-on, this project contributes to a more responsible way of evaluating computer vision systems.

## Project Scope

This is not a project about inventing a new distracted driving architecture. Instead, it focuses on auditing an existing deep learning pipeline.

The project includes:

1. dataset preparation using the State Farm distracted driving dataset  
2. VGG16 fine-tuning for 10-class behavior classification  
3. validation-based performance evaluation  
4. Grad-CAM generation for visual interpretability  
5. subgroup-based qualitative comparison of attention patterns  

The project is therefore best understood as a **Responsible AI computer vision audit**, not only a classification benchmark.

