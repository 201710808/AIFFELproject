# Code Peer Review Templete
- 코더 : 최지호
- 리뷰어 : 이효준

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

  
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?

```python
# 수축 경로와 전환 구간에서 사용할 block
def contracting_block(channels, inputs, dropout=False, maxpool=True):
    conv = Conv2D(channels, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv =  Conv2D(channels, 3, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    if dropout: # 전환 구간에서 dropout을 사용하기 위함
        conv = Dropout(0.5)(conv)
    if maxpool: # 전환 구간에서 maxpooling을 사용하지 않기 위함
        pool = MaxPooling2D((2, 2))(conv)
        return conv, pool
    return conv

# 확장 경로에서 사용할 블럭
def Expanding_block(channels, inputs, conv):
    up = Conv2DTranspose(channels, 2, strides=(2, 2), padding="same")(inputs)
    merge = Concatenate(axis=-1)([conv, up])
    up = Conv2D(channels, 3, padding="same", kernel_initializer='he_normal')(merge)
    up = BatchNormalization()(up)
    up = Activation("relu")(up)
    up = Conv2D(channels, 3, padding="same", kernel_initializer='he_normal')(up)
    up = BatchNormalization()(up)
    up = Activation("relu")(up)
    return up
```
> U-Net의 _수축/확장_ 경로에 대한 주석처리와 가독성 좋게 작성되어 있어, 금방 이해가 됐습니다.

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?
> 아니요.

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
```python
from tensorflow.keras.utils import plot_model

# 모델 그래프를 그리기 위해 plot_model 함수를 사용합니다.
plot_model(unet_pp_model, to_file='model.png', show_shapes=False, show_layer_names=False)
```
> 참고할 자료와 모델 flowchart 그림을 통해 제대로 구현 및 이해됐음을 확인할 수 있었습니다.

> U-Net++ 구현에서 수축과 확장블록을 잘 정리해 놓은 내용이 인상적이었습니다.

- [⭕] 5.코드가 간결한가요?
```python
for i in range(1, num_images + 1):
    image_path = dir_path + f'/image_2/00{str(i).zfill(4)}_10.png'
    label_path = dir_path + f'/semantic/00{str(i).zfill(4)}_10.png'

    # unet_model 예측 결과 얻기
    unet_output, unet_prediction, unet_target = get_output(
        unet_model,
        test_preproc,
        image_path=image_path,
        output_path=dir_path + f'./unet_result_{str(i).zfill(3)}.png',
        label_path=label_path
    )

    # unet_pp_model 예측 결과 얻기
    unet_pp_output, unet_pp_prediction, unet_pp_target = get_output(
        unet_pp_model,
        test_preproc,
        image_path=image_path,
        output_path=dir_path + f'./unet_pp_result_{str(i).zfill(3)}.png',
        label_path=label_path
    )

    # IOU 스코어 계산
    unet_iou = calculate_iou_score(unet_target, unet_prediction)
    unet_pp_iou = calculate_iou_score(unet_pp_target, unet_pp_prediction)

    # 이미지 결과 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # unet_model 결과
    axes[0].imshow(unet_output)
    axes[0].set_title("unet_model")
    axes[0].text(0, unet_output.height, f"IOU: {unet_iou:.4f}", color='white', backgroundcolor='black')

    # unet_pp_model 결과
    axes[1].imshow(unet_pp_output)
    axes[1].set_title("unet_pp_model")
    axes[1].text(0, unet_pp_output.height, f"IOU: {unet_pp_iou:.4f}", color='white', backgroundcolor='black')


    plt.show()
```
> 여러 이미지를 비교하기에 좋은 코드로 간결하고 잘 배웠습니다.!
