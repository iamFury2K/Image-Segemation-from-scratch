
# UNet Segmentation Project

## Description
This project implements a UNet architecture for image segmentation tasks. It includes a custom dataset loader for handling images and masks, utilities for model training, checkpointing, and evaluation, as well as the UNet model itself. The model is designed to be trained on a specific set of images and their corresponding segmentation masks to learn to perform segmentation tasks effectively.

## Installation
Clone the repository and install the required dependencies.

```
git clone https://github.com/yourusername/unet-segmentation.git
cd unet-segmentation
pip install -r requirements.txt
```

## Structure
- `dataset.py`: Module to handle image and mask loading.
- `architecture/UNET.py`: Contains the UNet model implementation along with a double convolution block.
- `utils/utils.py`: Utilities for training including loaders, checkpointing, accuracy checking, and prediction saving.
- `train.py`: Script to execute the training process including setting up data loaders, model, and training loops.

## Usage
To start training the UNet model, run the following command:

```bash
python train.py
```

## Features
- Image and mask dataset handling.
- Customizable UNet architecture.
- Model checkpointing.
- Training and validation accuracy evaluations.
- Saving predictions as images.

## Contributing
Contributions are welcome. Please fork the project and submit a pull request.

## License
Specify your project license here, commonly MIT or GPL-3.0.

## Authors
- **Muneeb Mushtaq** - Initial work - UNet from scratch (https://github.com/iamFury2K)

## Acknowledgments
- Thanks to the developers of the PyTorch library.
- Inspired by the original UNet paper.
```

