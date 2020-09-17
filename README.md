torchemotion: an emotion library for PyTorch
========================================

The aim of torchemotion is to apply [PyTorch](https://github.com/pytorch/pytorch) and [torchaudio](https://github.com/pytorch/audio) to the emotion recognition domain. We begin with providing basic dataloaders to read popular emotional datasets.

- [Dataloaders for common emotional speech datasets](https://github.com/alanwuha/torchemotion/tree/master/datasets)

## Dataloaders 

[Dataloaders](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=dataloader) for the following emotional datasets are currently available:

- [IEMOCAP - Interactive Emotional Dyadic Motion Capture](https://sail.usc.edu/iemocap/)
- [EmoDB - Berlin Database of Emotional Speech](http://emodb.bilderbar.info/start.html)
- [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976#.X2OIfnUzbJw)

## Example Usage

<details open>
<summary>EmoDB</summary>

First, download the EmoDB dataset from [here](http://emodb.bilderbar.info/docu/#download). 

Then, initialize an `EmodbDataset` object by passing in the path to the `download` directory that resides in the `emodb` directory:

```python
import EmodbDataset

# Initialize EmodbDataset
emodb_data = EmodbDataset('/emodb/download')

# Iterate over data
for index, sample in enumerate(emodb_dataset):
    print(i, sample)
```
</details>

## Dependencies

- pytorch
- torchaudio
- pandas
- numpy

## Disclaimer on Datasets

This library does not host or distribute these dataset, or claim that you have license to use the dataset. It is your responsiblity to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a GitHub issue. Thanks for your contribution to the speech emotion recognition community!
