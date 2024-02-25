# Segmentation-based BI-RADS Classification of Breast Tumours in Ultrasound Images

This repository contains the code accompanying the paper "Segmentation-based BI-RADS Classification of Breast Tumours in Ultrasound Images." Our work presents a comprehensive approach to classifying breast tumors in ultrasound images based on the BI-RADS system, leveraging a segmentation model to enhance the classification performance.

## Getting Started

### Prerequisites

Before running the code, ensure that you have a suitable Python environment with necessary dependencies installed. You can install the dependencies via:

```bash
pip install -r requirements.txt
```

### Setup

1. **Download the Segmentation Model**: First, you need to download the pre-trained segmentation model. Click [here](https://drive.google.com/file/d/1bFXnCTWLjMJdPV25lamli-NMCK5HDo31/view?usp=sharing) to download, and then place the model into the `checkpoints` folder within this repository.

2. **Prepare Your Dataset**:
   - Place your ultrasound images in the `images/full_image` folder. Organize the images into separate folders for each dataset.
   - Use the `generate_dataset.py` script to generate a masked dataset from your images.
   - Prepare the dataset with cut tumor images by running the `cut_borders.py` script.

### Training

- To train a binary benign/malignant classifier, use the following command:

  ```bash
  python train_classifier.py
  ```

- To train a BI-RADS classifier, run:

  ```bash
  python train_birads.py
  ```

It is recommended to train three models for: full images, masked images, and images with cut borders.

### Validation

After training the models, use the `multimodal_validation.py` script to compare the performance of the models and their ensemble:

```bash
python multimodal_validation.py
```

This script facilitates the comparison of single-modal and multi-modal approaches for breast tumor classification in ultrasound images.


## Citation

If you find this work useful in your research, please consider citing our paper:

```
Paper to be published
```


