# Code Peer Review Templete
- 코더 : 최지호
- 리뷰어 : 장승우

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

네, 100점 처리된 것으로 정상 동작 확인했어요~
  
- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?

네, 주석을 잘 달아줘서 이해가 쉬웠어요~

```
    def _match_anchor_boxes( # 앵커박스와 실제 객체의 바운딩 박스 간의 매칭 수행
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou) # max_iou가 0.5보다 클 경우 positive_mask
        negative_mask = tf.less(max_iou, ignore_iou) # max_iou가 0.4보다 작을 경우 negative_mask
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask)) # positive도 negative도 아닐 경우 ignore
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

```

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?

아니요, 에러나 경고가 발생하지 않았어요~

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

네, 직접 로직을 작성해서 등록했고, bbox 윤곽선 이미지도 빨간색으로 변경했어요~

```
    
    person = ['Pedestrian', 'Person_sitting']
    vehicle = ['Car', 'Van', 'Truck', 'Cyclist']
    
    if any(p in person for p in class_names): # 검출된 물체에 사람이 포함된다면
        print("Stop!")
        return "Stop"
    
    elif any(v in vehicle for v in class_names): # 검출된 물체에 차량이 포함된다면
        print("Well...")
        for [x_min, y_min, x_max, y_max] in detections.nmsed_boxes[0]:
            if all(coord == 0.0 for coord in [x_min, y_min, x_max, y_max]): # [[검출된 박스 좌표], [0, 0, 0, 0]]순으로 데이터가 저장됨
                print("Go!")                     # 따라서 [0, 0, 0, 0]차례가 오면 바로 for문을 종료시켜 쓸데없이 for문을 추가로 돌리지 않게 함
                return "Go" # 300픽셀 이상 크기의 차량이 없다면
            
            h = y_max - y_min # 검출된 물체의 세로
            w = x_max - x_min # 검출된 물체의 가로
            if h >= 300 or w >= 300: # 가로 혹은 세로가 300픽셀 이상이라면
                print("Stop!")
                return "Stop"
            
    else: # 사람도 차량도 없다면    
        print("Go~")
        return "Go"
```

- [⭕] 5.코드가 간결한가요?

네, 복잡하지 않고 간결하게 작성했어요~

# 참고 링크 및 코드 개선 여부

개념 설명을 이미지 자료로 설명 및 테스트 중 발생하는 물음에 대해 답변도 되어있어서 좋았어요~
