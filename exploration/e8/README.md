아이펠캠퍼스 온라인4기 피어코드리뷰 []

- 코더 : 최지호
- 리뷰어 : 이성주

----------------------------------------------

**PRT(PeerReviewTemplate)**

** [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
	*  1. 다양한 방법으로 Text Classification 태스크를 성공적으로 구현하였다
		+ 3가지 모델을 시도 하였고 모두 성공 하였습니다.
	* 2. gensim을 활용하여 자체학습된 혹은 사전학습된 임베딩 레이어를 분석하였다.
		+ gensim을 이용하여 분석을 하였습니다.
	* 3. 한국어 Word2Vec을 활용하여 가시적인 성능향상을 달성했다.
		+ 1번 모델 Accuracy: 0.831 = > 0.851
		+ 2번 모델 Accuracy: 0.844 => 0.849
		+ 3번 모델 Accuracy: 0.708 => 0.852
		+ 성능이 크게 향상 되었습니다.

** [o] 주석을 보고 작성자의 코드가 이해되었나요?
	* print(weights.shape)    # shape: (vocab_size, embedding_dim)
	- 주석을 보고 코드의 이해가 더 쉬워졌습니다.
** [x] 코드가 에러를 유발할 가능성이 있나요?
	* 에러를 유발할 가능성이 없습니다.	
** [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
	* 인터뷰를 통해 코드를 잘 이해하고 코드를 작성 하였습니다.

** [o] 코드가 간결한가요?
	* ``` index_to_word = {index:word for word, index in word_to_index.items()} ``` 등과 같이 파이써닉 하게 코드작성을 하였습니다.

----------------------------------------------

참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.

* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
``` python
word_to_index["<PAD>"] = 0
word_to_index["<BOS>"] = 1
word_to_index["<UNK>"] = 2  # unknown
word_to_index["<UNUSED>"] = 3
```
해당 부분을 load_data 함수에서 같이 하면 더 좋았을 것 같습니다.
``` python

def load_data(train_data, test_data, num_words=10000):
    # 데이터의 중복 제거
    train_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)

    # NAN 결측치 제거
    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')

    # 한국어 토크나이저로 토큰화 및 불용어 제거
    X_train = []
    for sentence in train_data['document']:
        temp = tokenizer.morphs(sentence)
        temp = [word for word in temp if not word in stopwords]
        X_train.append(temp)

    X_test = []
    for sentence in test_data['document']:
        temp = tokenizer.morphs(sentence)
        temp = [word for word in temp if not word in stopwords]
        X_test.append(temp)
    
    # 사전word_to_index 구성
    words = np.concatenate(X_train).tolist()    # 리스트로 변환
    counter = Counter(words)    # 단어의 등장 횟수를 가진 리스트
    counter = counter.most_common(num_words-4)  # 내림차순 정렬
                                            # 상위 num_words개 단어 반한
                                            # 앞의 4개는 비어있는 문자열로 대체
    vocab = ['<PAD>', '<BOS>', '<UNK>', '<UNUSED>'] + [key for key, _ in counter] # 해당 부분으로 수정 하면 더 깔끔한 코드가 되었을 것 같습니다.
    word_to_index = {word:index for index, word in enumerate(vocab)}

    # 텍스트 스트링을 사전 인덱스 스트링으로 변환
    def wordlist_to_indexlist(wordlist):
        return [word_to_index[word] if word in word_to_index
                else word_to_index[''] for word in wordlist]
    
    X_train = list(map(wordlist_to_indexlist, X_train))
    X_test = list(map(wordlist_to_indexlist, X_test))

    # X_train, y_train, X_test, y_test, word_to_index 리턴
    return X_train, np.array(list(train_data['label'])),\
            X_test, np.array(list(test_data['label'])), word_to_index
```
