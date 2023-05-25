아이펠캠퍼스 온라인4기 피어코드리뷰 []

코더 : 최지호
리뷰어 : 김창완
PRT(PeerReviewTemplate)

코드가 정상적으로 동작하고 주어진 문제를 해결했나요? - o  
    - 인물모드, 아웃포커싱, 배경까지 완벽하게 완수하셨습니다

주석을 보고 작성자의 코드가 이해되었나요?
    - 주석을 많이 달지는 않으셨지만 관련 링크 첨부로 어느정도 이해가 되었습니다  
    - 주석이 영어라 멋있었어요

코드가 에러를 유발할 가능성이 있나요? - x  
    - 에러는 아무리 돌려도 안나왔습니다

코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기) - o  
```python
def change_bg(img, output, img_bg, blur, index):
    colormap = np.zeros((256, 3), dtype = int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    colormap[:20]
    
    r, g, b = colormap[index]
    seg_color = (b,g,r)

    seg_map = np.all(output==seg_color, axis=-1) 
    print(seg_map.shape)

    img_mask = seg_map.astype(np.uint8) * 255
    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    
    img_concat = np.where(img_mask_color==255, img, img_bg)

    return img_concat

img_result = change_bg(img_resize, output, img_bg_rgb, 20, index)
plt.imshow(img_result)
```
- 이 부분 이해가 안됐으나 질문 후 잘 답변 해주셨습니다

코드가 간결한가요? - o  
    - 제가 더이상 간결하게는 못만들거 같습니다

참고 링크 및 코드 개선  
```python
def make_bg_blur(img, output, index, blur):
    colormap = np.zeros((256, 3), dtype = int)
```
이 부분에서 변수안에 언더바("_")를 넣어서 변수 충돌을 막을 수 있습니다
