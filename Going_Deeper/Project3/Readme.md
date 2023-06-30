## Aiffel_peer_review(6/30)
## Team : 최지호
## Reviewer : 최우정
-----------------------------------------------------------------------
## I review project 6 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Going deeper Project 3
- Dataset : Class Activation Map 만들기 
- Problem : 

### STep1 : Preprocessing & CAM 구현 -> 기존 train, test normalize good!

```python
   Data Load, pre processing 

   def normalize_and_resize_img(input):
    # Normalizes images: `uint8` -> `float32`
    image = tf.image.resize(input['image'], [224, 224])
    input['image'] = tf.cast(image, tf.float32) / 255.
    return input['image'], input['label']

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(
        normalize_and_resize_img,
        num_parallel_calls=2
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```


### Model performance visualization -> 학습결과를 Loss, Accuracy 시각화하여 체크 good!
![image](![image](https://github.com/201710808/AIFFELproject/assets/127918850/2240976f-bcb0-4c3c-9566-52351222890c)


### CAM 생성 함수 만들기 (추가 공부) -->

### CAM  시각화
직접 코드를 분석하며 'np.clip으로 범위를 지정하지 않으면 어떻게 되는지 결과를 살펴보신  점도 좋았습니다. 

![image](![image](![image](https://github.com/201710808/AIFFELproject/assets/127918850/46acac2a-4aa9-4162-a311-01ca0baa4f6d)



### Conclusion

CAM에 대해 정확하게 이해를 가
의문점에 대해서도 하단과 같이 잘 정리 해주셨고, 코드도 여러 방향으로 시도해보신 모습이 좋았습니다. 

##(추정)GAP를 수행하기 전, class의 숫자와 GAP에 들어가는 이미지의 채널 수를 맞춰주면 전반적인 성능이 개선되는 것 같습니다.


### 1,2,3,4,5 pass (O)  
-----------------------------------------------------------------------
