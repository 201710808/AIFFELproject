# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 최지호
- 리뷰어 : 부석경

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

1. **Text recognition을 위해 특화된 데이터셋 구성이 체계적으로 진행되었다.**  
  네.
```python
batch_indicies = self.index_list[
            idx*self.batch_size:
            (idx+1)*self.batch_size
        ]
```
```python
batch_indicies = self.index_list[
            idx*self.batch_size:
            (idx+1)*self.batch_size
        ]
```
```python
def ctc_lambda_func(args): # CTC loss를 계산하기 위한 Lambda 함수
    labels, y_pred, label_length, input_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
```

2. **CRNN 기반의 recognition 모델의 학습이 정상적으로 진행되었다.**
  네, 모델 훈련을 체계적으로 진행했습니다.
![image](https://github.com/201710808/AIFFELproject/assets/71332005/a6940a39-5b1a-45d4-b959-a7b385e8c185)
![image](https://github.com/201710808/AIFFELproject/assets/71332005/1a4378f2-f883-4ee4-b869-1bd8b7017e6f)

3. **keras-ocr detector와 CRNN recognizer를 엮어 원본 이미지 입력으로부터 text가 출력되는 OCR이 End-to-End로 구성되었다.**
```
HOME_DIR = os.getenv('HOME')+'/aiffel'
SAMPLE_IMG_PATH = HOME_DIR + '/Going_Deeper/sample2.jpg'

img_pil, cropped_img = detect_text(SAMPLE_IMG_PATH)
show_images(img_pil, cropped_img)

for _img in cropped_img:    
    recognize_img(model, _img)
```
![image](https://github.com/201710808/AIFFELproject/assets/71332005/b2a901d4-fe2f-41b5-9aa8-d2b87ff01343)   
![image](https://github.com/201710808/AIFFELproject/assets/71332005/555a55f7-4dc7-45da-a614-49da89b247e4)   
이에 따른 분석과 추가 실험을 진행했습니다.
   
### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**  
  네,
```python
# 문자 인식 함수
def detect_text(img_path):
    # cv2로 이미지 불러오기
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_batch = np.expand_dims(img_rgb, axis=0) # 배치 차원 추가
    
    # keras ocr 수행
    ocr_result = detector.detect(img_batch)[0] # 첫 번째 결과만 가져오기
    
    # 시각화를 위해 ImageDraw 사용
    img_pil = Image.fromarray(img_rgb)
    result_img = img_pil.copy()
    img_draw = ImageDraw.Draw(result_img)
    
    # 인식된 부분만 잘라낸 이미지들을 저장할 리스트
    cropped_imgs = []
    
    # 이미지 잘라내기
    for text_result in ocr_result:
        img_draw.polygon(text_result, outline='red')
        x_min = text_result[:,0].min() - 5
        x_max = text_result[:,0].max() + 5
        y_min = text_result[:,1].min() - 5
        y_max = text_result[:,1].max() + 5
        word_box = [x_min, y_min, x_max, y_max]
        cropped_imgs.append(img_pil.crop(word_box))
        
    return result_img, cropped_imgs

```

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
네, 모델에 관해서 이야기를 했습니다.    
![image](https://github.com/201710808/AIFFELproject/assets/71332005/c5702b1f-e519-4c5d-b5b2-dce4b4769642)

https://tw0226.tistory.com/90   
위 링크를 참고해 주었고 설명해주어 더욱 잘 이해가 쉬었습니다.

### **[⭕] 코드가 간결한가요?**
네. 복잡한 코드를 찾지 못했습니다.


----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
