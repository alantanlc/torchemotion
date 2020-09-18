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
<summary>IEMOCAP</summary>

First, obtain the IEMOCAP dataset by filling out the electronic release form and submit a request [here](https://sail.usc.edu/iemocap/iemocap_release.htm).

Then, initialize an [IemocapDataset](./datasets/IemocapDataset.py) object by passing in the path to the __IEMOCAP_full_release__ directory.

```python
import IemocapDataset

# Initialize IemocapDataset
iemocap_dataset = IemocapDataset('./IEMOCAP_full_release')

# Iterate over data
for index, sample in enumerate(iemocap_dataset):
    print(index, sample)
```
</details>

<details open>
<summary>EmoDB</summary>

First, download the EmoDB dataset from [here](http://emodb.bilderbar.info/docu/#download). 

Then, initialize an [EmodbDataset](./datasets/EmodbDataset.py) object by passing in the path to the __download__ directory.

```python
import EmodbDataset

# Initialize EmodbDataset
emodb_data = EmodbDataset('./download')

# Iterate over data
for index, sample in enumerate(emodb_dataset):
    print(index, sample)
```
</details>

<details open>
<summary>RAVDESS</summary>

First, download the RAVDESS dataset from [here](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1).

Then, initialize an [RavdessDataset](./datasets/RavdessDataset.py) object by passing in the path to the __Audio_Speech_Actors_01-24__ directory.

```python
import RavdessDataset

# Initialize RavdessDataset
ravdess_dataset = RavdessDataset('./Audio_Speech_Actors_01-24')

# Iterate over data
for index, sample in enumerate(ravdess_dataset):
    print(index, sample)
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
