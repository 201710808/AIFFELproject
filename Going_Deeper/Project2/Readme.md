## Aiffel_peer_review(6/28)
## Team : 최지호
## Reviewer : 김신성
-----------------------------------------------------------------------
## I review project 6 following above rules
- 1.Did the code work properly and fix the given issue?
- 2.Did I look at the comments and understand the author's code? And it is suitable?
- 3.Is there a possibility that the code will cause an error?
- 4.Did the code writer understand and write the code correctly?
- 5.Is the code concise and expandable?
- 6.etc
-----------------------------------------------------------------------
## Going deeper Project 2
- Dataset : Official Image net dataset
- Problem : Implement various data augmentation methods

### STep1 : Preprocessing 

   Data Load, pre processing(Drop missing data, Non-unique data, tokenize etc...)
   
### Data preprocessing and augmentation
```python
def normalize_and_resize_img(image, label):
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 1)
    return image, label

def onehot(image, label, num_classes=120):
    label = tf.one_hot(label, num_classes)
    return image, label

def apply_normalizer_on_dataset(ds, is_test=False, batch_size=16,
                                with_aug=False, with_cutmix=False, with_mixup=False):
    ds = ds.map(
        normalize_and_resize_img,
        num_parallel_calls=2
    )
    if not is_test and with_aug:
        ds = ds.map(
            augment
        )
    ds = ds.batch(batch_size)
    if not is_test and with_cutmix:
        ds = ds.map(
            cutmix,
            num_parallel_calls=2
        )
    elif not is_test and with_mixup:
        ds = ds.map(
            mixup,
            num_parallel_calls=2
        )
    else:
        ds = ds.map(
            onehot,
            num_parallel_calls=2
        )
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

(ds_train, ds_test), ds_info = tfds.load(
    'stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,
    shuffle_files=True,
    with_info=True,
)
ds_train_no_aug = apply_normalizer_on_dataset(ds_train, with_aug=False) # 원래 훈련 데이터
ds_train_aug = apply_normalizer_on_dataset(ds_train, with_aug=True) # 기본적인 Augmentation 적용
ds_train_aug_cutmix = apply_normalizer_on_dataset(ds_train, with_aug=True, with_cutmix=True)    # 기본 + CutMix 적용
ds_train_aug_mixup = apply_normalizer_on_dataset(ds_train, with_aug=True, with_mixup=True)  # 기본 + Mixup 적용
ds_test = apply_normalizer_on_dataset(ds_test, is_test=True)    # 테스트 데이터
```
"Data augmentation and normalization were well implemented and applied." => good!


# Verg good point!!
```python
# 전체 Text 데이터에 대한 전처리 : 10분 이상 시간이 걸릴 수 있습니다. 
clean_text = []
# 인덱스 초기화
data.reset_index(drop=True, inplace=True)

# [[YOUR CODE]]
for i in range(0, len(data)):
    sen_data = preprocess_sentence(data["text"][i])
    clean_text.append(sen_data)
    if i / 10000 in list(range(1,10)):
        print(f"{i}번 째 반복중")
# 전처리 후 출력
print("Text 전처리 후 결과: ", clean_text[:5])
```

### Model performance visualization
```python
# 결과 시각화
import pickle
import matplotlib.pyplot as plt

with open('data/epoch20/history_resnet50_no_aug.pickle', 'rb') as file:
    history = pickle.load(file)
no_aug_val_loss = history['val_loss']
no_aug_val_acc = history['val_accuracy']

epochs = range(1, len(no_aug_val_loss) + 1)

with open('data/epoch20/history_aug.pickle', 'rb') as file:
    history = pickle.load(file)
aug_val_loss = history['val_loss']
aug_val_acc = history['val_accuracy']

with open('data/epoch20/history_resnet50_aug_cutmix.pickle', 'rb') as file:
    history = pickle.load(file)
aug_cutmix_val_loss = history['val_loss']
aug_cutmix_val_acc = history['val_accuracy']

with open('data/epoch20/history_aug_mixup.pickle', 'rb') as file:
    history = pickle.load(file)
aug_mixup_val_loss = history['val_loss']
aug_mixup_val_acc = history['val_accuracy']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, no_aug_val_loss, 'r', label='No Augmentation')
plt.plot(epochs, aug_val_loss, 'b', label='Augmentation')
plt.plot(epochs, aug_cutmix_val_loss, 'g', label='Aug + CutMix')
plt.plot(epochs, aug_mixup_val_loss, 'purple', label='Aug + Mixup')
plt.title('Model Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.ylim(0.75, 1.75)

plt.subplot(1, 2, 2)
plt.plot(epochs, no_aug_val_acc, 'r', label='No Augmentation')
plt.plot(epochs, aug_val_acc, 'b', label='Augmentation')
plt.plot(epochs, aug_cutmix_val_acc, 'g', label='Aug + CutMix')
plt.plot(epochs, aug_mixup_val_acc, 'purple', label='Aug + Mixup')
plt.title('Model Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim(0.55, 0.80)

plt.tight_layout()
plt.show()
  ```

![image](https://github.com/201710808/AIFFELproject/assets/91248817/ab9bc0db-8011-4c44-b71a-54e4990e29e2)

"I think it would be better to look for improvement measures that can lead to performance like the paper and add additional research results"

### Conclusion

전체적으로 모델을 save하고 augmentation 기법을 적용하여 실제 모델에 훈련해 보는 것 까지 전체적인 연구의 흐름을 잘 알고있는것 같습니다.
훈련 결과에 대한 시각화와 거기에 대한 분석을 같이 첨부해주셔서 한결 읽기 편했습니다.
본 주제에서 augmentation의 개선방안으로 주어진 cut mix 등의 방법을 적용했을때 어떻게 하면 더 좋은 performance를 이끌어낼지 생각해보고 추가적인 연구 결과를 덧붙혀 추가한다면 훨씬 더 좋은 프로젝트가 될 것 같습니다!
수고하셨습니다!!


## 1,2,3,4,5 pass (O)  
-----------------------------------------------------------------------
