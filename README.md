# Computer Vision Project 2 - AIRO
**Joint Detection of AI-Generated Images and Post-Processing Alterations**

**Authors**: Alessandro Orlando 2038123, Jacopo Stizza 2086730

**Note**: see the Notebook text/output-cells and the PowerPoint report for the complete analysis

**Dataset Requirement**:

1. download the dataset (25GB) from https://zenodo.org/records/14963880

2. unzip "RRDataset_original_train_val.tar.gz" and "RRDataset_final.tar.gz"

3. put them in a folder named as "Dataset"

## 1. Project goal

This project studies a practical forensic problem in computer vision: given a single image, can we infer both whether it is a genuine photograph or AI-generated, and which post-processing transformation it has undergone?

The notebook implements a unified multi-task learning pipeline that produces two predictions from the same shared representation. The first task is binary real/fake detection. The second task is transformation classification over the three RRDataset regimes: original, internet-transmitted, and re-digitized.

In mathematical form, for an input image $x$ the model predicts two labels:

$$
\hat{y}_{rf} = f_{rf}(x), \qquad \hat{y}_{tr} = f_{tr}(x)
$$

where $y_{rf} \in \{0, 1\}$ and $y_{tr} \in \{0, 1, 2\}$. The joint training objective is a weighted sum of two cross-entropy losses:

$$
\mathcal{L} = \lambda_{rf} \mathcal{L}_{rf} + \lambda_{tr} \mathcal{L}_{tr}
$$

## 2. Dataset

The project uses RRDataset from the Zenodo benchmark "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios". The benchmark is relevant because it models the kind of degradations that appear in realistic online circulation.

## 3. Pipeline

Imports. Dependencies and vision libraries.

Globals. Paths, seeds, hyperparameters, and run flags.

Utils. Archive extraction, label inference, balanced sampling, and helper routines.

Data. Dataset indexing, split construction, transformations, loaders, and sample visualization.

Network. Shared CNN backbone with two task-specific classification heads.

Train. Baseline runs, joint multi-task optimization, and optional ablation over loss weights.

Evaluation. Overall metrics, classification reports, confusion matrices, and cross-class analysis.

Plots & Reporting

## 4. Methodology

The model is built around a pretrained ResNet backbone. The backbone extracts a shared feature vector $f \in \mathbb{R}^d$ from the input image. Two independent linear heads then map the same representation to the two output spaces:

$$
z_{rf} = W_{rf} f + b_{rf}, \qquad z_{tr} = W_{tr} f + b_{tr}
$$

The corresponding class probabilities are obtained with a softmax layer. Training is performed with AdamW, which is well suited to transfer learning because it combines adaptive optimization with decoupled weight decay.

The project also includes unimodal baselines, implemented by setting one task weight to zero.

The metrics used mainly are accuracy, balanced accuracy, macro F1 and joint correctness.

## 5. Key idea

These lines summarize the architecture, the training objective, and the model-selection criterion.

```python
features = self.backbone(x)
real_logits = self.real_head(features)
transform_logits = self.transform_head(features)
```

```python
total_loss = w_real * loss_real + w_transform * loss_transform
```

```python
score = 0.5 * (val_metrics["real_acc"] + val_metrics["trans_acc"])
```

## 6. Outputs

- best_model.pth: checkpoint of the best validation model.
- history_*.csv: training curves for loss and accuracy.
- dataset_summary.txt: class and split statistics.
- samples_train.png, samples_val.png, samples_test.png: qualitative sanity-check figures.
- evaluation_summary.json: compact JSON summary of the final metrics.
- report_real_fake.txt and report_transform.txt: detailed classification reports.
- cross_class_accuracy.csv: real/fake accuracy broken down by transformation and class.
- redigital_method_accuracy.csv: optional per-method analysis for re-digitization variants.
- confusion_real_fake.png and confusion_transform.png: confusion matrices.
- figures/: polished plots intended for the written report.

## 6.1. Results

- real/fake accuracy: about 0.953
- transformation accuracy: about 0.833
- real/fake macro F1: about 0.95
- transformation macro F1: about 0.83

The binary detector is therefore already strong and balanced, while the transformation classifier is good but still more difficult, especially for the internet-transmitted class.

## 7. How to run

1. Open CV_OrlandoStizza_Project2.ipynb
2. Make sure the dataset is available under Dataset/ either as extracted folders or as the original archives.
3. Review the Globals cell and choose the desired execution mode.
4. Run the notebook top to bottom.

The notebook is ready to run as long as it is executed from the repository root and the folder structure is preserved.

The most relevant settings are RUN_FULL, SUBSET_PER_GROUP, USE_OFFICIAL_TEST, EPOCHS, BATCH_SIZE, and IMAGE_SIZE. For a full experiment, larger image size and more epochs are appropriate. For a fast test, the notebook can be reduced to a small balanced subset.

Suggested full-run values depend on the available hardware, but a reasonable starting point is:

```python
EPOCHS = 5
BATCH_SIZE = 16 
IMAGE_SIZE = 224
```

## References

Li, Chunxiao, et al. "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios." ICCV 2025.

Shao, Rui, Tianxing Wu, and Ziwei Liu. "Detecting and grounding multi-modal media manipulation." CVPR 2023.