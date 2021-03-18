# vggish
My implementation of VGGish. The model has an easy to use implementation and assimilates VGGish into the latest tensorflow version.

## Set-Up:

To clone the repository please use the command:
```console
git clone https://github.com/AmanPriyanshu/vggish.git
```

## Describing Dataset:

To load and describe the dataset we will first use the modules present within properties_of_dataset.py. Following is a simple implementation of its usage:

CODE:

```python
from vggish_loader import loading_dataset, properties_of_dataset, loading_wav
import numpy as np

PATH = "./PATH/TO/DATASET/DIRECTORY/"

if __name__ == '__main__':
  sr, wav_length = properties_of_dataset('./train/')
  
  print('Average SR:', sum(sr)/len(sr))
  print('Average WAV Length', sum(wav_length)/len(wav_length), '\n')

  print('Lowest SR', min(sr))
  print('Lowest WAV Length', min(wav_length), '\n')

  print('Highest SR', max(sr))
  print('Highest WAV Length', max(wav_length))

  print('Appropriate Steps would be SR * seg_len (Eg: seg_len=5):', 5*sum(sr)/len(sr))
```

OUTPUT:

```console
Average SR: 22050.0
Average WAV Length 56040.84209143458 

Lowest SR 22050
Lowest WAV Length 27653 

Highest SR 22050
Highest WAV Length 116247
Appropriate Steps would be SR * seg_len (Eg: seg_len=5): 110250.0
```

## Load Dataset:

To load the dataset we will be using the following example code segment.

```python
dataset_x, dataset_y, label_map = loading_dataset('./train/', 110000)
```

Where, dataset_x defines the input and dataset_y defines the respective label identified for it. The label_map gives us a dictionary to map the results to the appropriate labels.

## Contributions:

Contributions are welcome on this repository, if you are willing to submit or push any proposal please check out the contributions.md
