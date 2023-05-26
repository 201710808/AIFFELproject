아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 최지호
- 리뷰어 : 김창완

----------------------------------------------

** [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
    - 전처리와 추출적 요약, 표 만들기 등 완벽하게 task를 전부 수행 했습니다  

** [o] 주석을 보고 작성자의 코드가 이해되었나요?  
    - 모든 step에 주석이 달려있어 읽기 편했고 전체적인 flow가 이해 되었습니다.
```python
threshold = 5
total_cnt = len(tar_tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tar_tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 %s'%(total_cnt - rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
** [o] 코드가 에러를 유발할 가능성이 있나요?  
    - 검사결과 변수명, scope등에서 에러가 날만한건 찾지 못하였습니다.
  
** [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)  
    - 각 step, 특히 어텐션 메커니즘에서 질문시 이해가 뛰어났습니다.
```python
# 어텐션 층(어텐션 함수)
attn_layer = AdditiveAttention(name='attention_layer')

# 인코더와 디코더의 모든 time step의 hidden state를 어텐션 층에 전달하고 결과를 리턴
attn_out = attn_layer([decoder_outputs, encoder_outputs])


# 어텐션의 결과와 디코더의 hidden state들을 연결
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# 디코더의 출력층
decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

# 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.summary()
```

** [o] 코드가 간결한가요?  
    - 제 수준에서는 이것보다 더 간단하게 짜기는 힘들 것 같습니다
    간결하게 잘 짜셨습니다

----------------------------------------------

참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.