# Simple Gender Filter

Simple deep learning model which classify gender of person in input image

(Pretrained model must use aligned face)

![gender_filter/female.png](Simple%20Gender%20Filter%20e648b46fe30d4291ad6254d7e61847e3/female.png)

                                                          <score: 0.9987>

![gender_filter/male.png](Simple%20Gender%20Filter%20e648b46fe30d4291ad6254d7e61847e3/male.png)

                                                        <score: 5.0664e-07>

## How to Use

```python
from gender_filter import GenderFilter
GF = GenderFilter
output = GF(input)
```

- input : PIL image ‘RGB’color
- output: score (Male:1, Female:0)

# Self Training

## ready

- Make forder which have named ’{data_forder}’
- Make forder ‘F’ and ‘M’ in forder which made in ‘{data_forder}’
- Put Male image in forder ‘M’, and Female image in forder ’F’

## Training

```jsx
python train.py --data_path '{data_forder}'
```
