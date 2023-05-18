# Code Peer Review Templete
- 코더 : 최지호
- 리뷰어 : 이승한


# PRT(PeerReviewTemplate)

- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?
- [O] 3.코드가 에러를 유발한 가능성이 있나요?
- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
- [O] 5.코드가 간결한가요?


# 참고 링크 및 코드 개선 여부

1. learning rate 변화에 따른 y-test, y-prediction plot의 변화를 확인하면 더 좋은 인사이트를 얻지 않을까요?
```python
import matplotlib.pyplot as plt
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

for i in range(10):
    plt.scatter(df_X[:, i], df_y)
    plt.xlabel(f'X[{i}]')
    plt.ylabel(f'y')
    plt.show()
```

2. 저는 프로젝트2를 시간상 진행하지 못했는데, 전처리관련해서 꼼꼼하게 잘하신거 같습니다. 저도 같은 방식으로 적용해볼께요!
