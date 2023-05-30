아이펠캠퍼스 온라인4기 피어코드리뷰 [2023-05-30]

- 코더 : 최지호
- 리뷰어 : 이성주

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
|평가문항|상세기준|완료여뷰|
|-------|--------|-------|
|1. pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다.|데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.|5가지의 augmentation 적용 [코드 1]|
|2. pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다.|U-Net generator, discriminator 모델 구현이 완료되어 train_step의 output을 확인하고 개선하였다.|노드에 있는 U-Net generator, discriminator 모델 구현이 완료되었고, 학습 결과를 위해 epoch을 100으로 진행|
|3. 학습 과정 및 테스트에 대한 시각화 결과를 제출하였다.|10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.|epoch을 100으로 진행하여 10epoch보다 좋은 결과를 도출 하였습니다 ![image](https://github.com/201710808/AIFFELproject/assets/29011595/e655efba-f1f3-4936-b660-2d9840e0b661)|

[코드 1]
``` python
from tensorflow import image
from tensorflow.keras.preprocessing.image import random_rotation

@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(seg_img, picture):
    stacked = tf.concat([seg_img, picture], axis=-1)
    
    _pad = tf.constant([[30,30],[30,30],[0,0]])
    if tf.random.uniform(()) < .5:
        # 50퍼센트 확률로 reflection padding이 30픽셀의 padwith만큼 적용됩니다.
        padded = tf.pad(stacked, _pad, "REFLECT")
    else:
        # 50퍼센트 확률로 constant padding이 30픽셀의 padwith만큼 적용됩니다.
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.)

    # 256, 256, 6 크기를 가진 이미지를 임의로 잘라냅니다.
    out = image.random_crop(padded, size=[256, 256, 6]) 
    
    # 50퍼센트 확률로 horizontal_flip시킵니다.
    out = image.random_flip_left_right(out) 
    
    # 50퍼센트 확률로 vertical_flip시킵니다.
    out = image.random_flip_up_down(out)    
    
    #   50퍼센트 확률로 반시계 방향으로 90 * (1~3)도 회전시킵니다.
    if tf.random.uniform(()) < .5:
        degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        degree = tf.cast(degree / 90, tf.int32)
        out = image.rot90(out, k=degree)
    
    return out[...,:3], out[...,3:]   
   ```
   
** [o] 주석을 보고 작성자의 코드가 이해되었나요?
아래 주석으로 코드 이해가 쉽게 되었습니다.
 ``` python
 from tensorflow import image
from tensorflow.keras.preprocessing.image import random_rotation

@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(seg_img, picture):
    stacked = tf.concat([seg_img, picture], axis=-1)
    
    _pad = tf.constant([[30,30],[30,30],[0,0]])
    if tf.random.uniform(()) < .5:
        # 50퍼센트 확률로 reflection padding이 30픽셀의 padwith만큼 적용됩니다.
        padded = tf.pad(stacked, _pad, "REFLECT")
    else:
        # 50퍼센트 확률로 constant padding이 30픽셀의 padwith만큼 적용됩니다.
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.)

    # 256, 256, 6 크기를 가진 이미지를 임의로 잘라냅니다.
    out = image.random_crop(padded, size=[256, 256, 6]) 
    
    # 50퍼센트 확률로 horizontal_flip시킵니다.
    out = image.random_flip_left_right(out) 
    
    # 50퍼센트 확률로 vertical_flip시킵니다.
    out = image.random_flip_up_down(out)    
    
    #   50퍼센트 확률로 반시계 방향으로 90 * (1~3)도 회전시킵니다.
    if tf.random.uniform(()) < .5:
        degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        degree = tf.cast(degree / 90, tf.int32)
        out = image.rot90(out, k=degree)
    
    return out[...,:3], out[...,3:] 
   ```

** [o] 코드가 에러를 유발할 가능성이 있나요?
``` python
test_ind = 3

data_path = 'data/cityscapes/val/'
f = data_path + os.listdir(data_path)[test_ind]
seg_img, picture = load_img(f)
```
 -> os.listdir()은 순서를 보장하지 않아서 3.jpg을 의도 한것이라면, 의도와 다를수 있습니다.

** [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
![image](https://github.com/201710808/AIFFELproject/assets/29011595/68f459cc-1a8c-4f36-a8ee-b8d4c2e1f91c)

-> augmentation 하는 부분을 특히 잘 이해하였으며, GAN방식을 전반적으로 이해하고 코드를 작성하였습니다.

** [o] 코드가 간결한가요?
``` python
     for i, n_filters in enumerate(filters):
            stride = 2 if i < 3 else 1
            padding = False if i < 3 else True
            use_bn  = False if i == 0 or i == 4 else True
            act = True if i < 4 else False
            self.blocks.append(DiscBlock(n_filters, stride, padding, use_bn, act))
```
-> 조건부 표현식을 사용하여 코드를 간결하게 표현

----------------------------------------------

참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

``` act = True if i < 4 else False ```
해당 코드는 
``` act = i < 4 ``` 
이와 같이 더 편하게 작성 할수 있습니다.
