# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 최지호
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
1. multiface detection을 위한 widerface 데이터셋의 전처리가 적절히 진행되었다.
```python
def make_example(image_string, image_infos):
    for info in image_infos:
        filename = info['filename']
        width = info['width']
        height = info['height']
        depth = info['depth']
        classes = info['class']
        xmin = info['xmin']
        ymin = info['ymin']
        xmax = info['xmax']
        ymax = info['ymax']

    if isinstance(image_string, type(tf.constant(0))):
        encoded_image = [image_string.numpy()]
    else:
        encoded_image = [image_string]

    base_name = [tf.compat.as_bytes(os.path.basename(filename))]
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=base_name)),
        'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'classes':tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'x_mins':tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'y_mins':tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'x_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'y_maxes':tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded_image))
    }))
    return example
```
* Augmentation & jaccard
  `crop()`, `_resize()`, `_flip()`, `_pad_to_square()`, `_distort()`, `_intersect()`다음과 같은 함수를 정의하였습니다.

2. SSD 모델이 안정적으로 학습되어 multiface detection이 가능해졌다.
```
   1에포크당 약 2분 정도가 걸리네요.
이제와서 loss 그래프를 그리기 위해 history를 만들지 않은 것을 후회하게 됩니다...
```
- 저도...

3.  이미지 속 다수의 얼굴에 스티커가 적용되었다.
![image](https://github.com/201710808/AIFFELproject/assets/71332005/dc01e662-f64e-4287-8480-78853b5a3b6b)

> 결론
> 어떤 이미지 사이즈가 들어와도 일단 인식은 된다.
> 하지만 얼굴 인식이 잘 안되는 부분에 있어서 사진 얼굴 크기의 문제인지, 무언가가 가려서 문제인지, 얼굴을 옆으로 틀어서 문제인지는 알 수 없다.
> (손으로 얼굴을 가려도 인식이 되기도 하고 안되기도 하기 때문, 마찬가지로 얼굴을 옆, 위로 틀어도 인식이 되기도 하고 안되기도 함..)
> 학습을 할 때는 꼭 loss 그래프를 그리도록 한다.(loss를 확인하지 않아서 어떤 학습 결과가 적절히 학습된 결과인지를 알 수가 없음)

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
```python
def draw_sticker_on_face(img, boxes, classes, scores, box_index, class_list, sticker_path):
    img_height = img.shape[0]
    img_width = img.shape[1]

    x_min = int(boxes[box_index][0] * img_width)
    y_min = int(boxes[box_index][1] * img_height)
    x_max = int(boxes[box_index][2] * img_width)
    y_max = int(boxes[box_index][3] * img_height)
    
    # 바운딩 박스의 가로, 세로 길이
    box_width = x_max - x_min
    box_height = y_max - y_min
    
    # 바운딩 박스의 중앙 픽셀
    x_center = x_min + int(box_width / 2) 
    y_center = y_min + int(box_height / 2)
    
    # 스티커 불러와서 바운딩 박스의 가로세로 중 최솟값에 맞춰 resize
    sticker_size = min(box_width, box_height)
    sticker = cv2.imread(sticker_path)
    sticker = cv2.resize(sticker, (sticker_size, sticker_size))
    
    # 스티커를 붙일 영역 설정
    sticker_area = img[int(y_center - sticker_size/2):int(y_center + sticker_size/2),
                       int(x_center - sticker_size/2):int(x_center + sticker_size/2)]
    
    # 스티커를 바운딩 박스의 중앙에 붙이기
    sticker_mask = (sticker != 255).any(axis=-1)
    img[int(y_center - sticker_size/2):int(y_center + sticker_size/2),
        int(x_center - sticker_size/2):int(x_center + sticker_size/2)][sticker_mask] =\
        sticker[sticker_mask]
    
    plt.imshow(img)
```

네, 위와같이 주석으로 코드를 잘 설명해 주었습니다.

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
 찾지 못했습니다.
 
### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
 실행한 이미지 증강 기법과 jaccard에 대해 질문했습니다. 
코드를 보며 설명해 주었습니다.

### **[⭕] 코드가 간결한가요?**
네 복잡한 코드를 느끼지 못했습니다.
![image](https://github.com/201710808/AIFFELproject/assets/71332005/ef21bef3-3e44-4128-8219-e83da42f54d7)


----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
