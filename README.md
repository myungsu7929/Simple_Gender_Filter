# Simple Gender Filter

Simple deep learning model predict gender of given image
(Pretrained model used aligned image)
               
![](./assets/grid.png)


# Self Training

## ready

- Make forder which have named ’{data_forder}’
- Make forder ‘F’ and ‘M’ in forder which made in ‘{data_forder}’
- Put Male image in forder ‘M’, and Female image in forder ’F’

## Training

```jsx
python train.py --data_path {data_forder}
```
