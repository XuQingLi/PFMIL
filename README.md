# PFMIL: Training and Feature Extraction Pipeline

## 1. Environment Setup
```bash
conda env create -f PFMIL_env.yml
conda activate PFMIL_env
```

## 2. Training and Feature Extraction
### 2.1 Stage 1 – Iterative fine-tuning of the feature extractor
This stage optimizes the feature extractor through a three-phase iteration to generate more discriminative instance features.

Step 1: Initial Feature Extraction
Generate instance embeddings using the initial ResNet50 weights.

```bash
CUDA_VISIBLE_DEVICES=[your_available_device_ids] \
python3 -u step0_patch2feature.py \
> step0_patch2feature_[method_name].log 2>&1 &
```

Stage 2 – Final training using the optimized features
Step 2: Train the Bag-level Classifier
Train a bag-level classifier using features extracted from Step 1.
```bash
CUDA_VISIBLE_DEVICES=[your_available_device_ids] \
python3 -u step1_train_classifier_[method_name].py \
> step1_train_classifier_[method_name].log 2>&1 &
```
Step 3: Feature Network Fine-tuning (ICMIL Distillation)
Use the ICMIL teacher–student distillation to fine-tune the feature extractor.
```bash
CUDA_VISIBLE_DEVICES=[your_available_device_ids] \
python3 -u step2_teacher_student_[method_name]_distillation.py \
> step2_teacher_student_[method_name]_distillation.log 2>&1 &
```
### 2.2 Stage 2: Final Training
```python
python trains.py
```
## 3. Testing

Evaluate the trained model on the test set.

```bash
python test.py
```
## 4. Visualization

Visualize feature distributions using t-SNE dimensionality reduction to understand feature separability.

```bash
python t-sne.py
```
