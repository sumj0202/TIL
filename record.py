# 6월 18일(수) - 오전 -
'''
머신러닝을 이용한 영어 텍스트 감성 분석
CountVectorizer 함수를 이용하여 임베딩
'''
### CountVectorize 모델 임폴트
from sklearn.feature_extraction.text import CountVectorizer

# 함수 호출 및 모델(객체) 생성
ko_cv = CountVectorizer()
en_cv = CountVectorizer(stop_words='english')

# 한글은 cv에 불용어가 없음

# 단어 사전 생성
cv_vocab = cv.vocabulary_
'''
-> text 데이터를 토대로 토큰화된 단어 사전 생성
-> dict 구조 key에는 단어 value에는 인덱스
'''
# 임베딩 행렬 생성
matrix = cv.transform('textdata').toarray()

'''
참조
한글 같은 경우에는 kiwi 형태소 분석기를 통해서
kiwipiepy.utils에서 stopwords 임폴트.
kw.tokenize(sent, stopwords=stop_words)
그 후, fit함수로 단어 사전생성
'''

### 절차 정리
'''
CountVectorizer 모델 임폴트
모델 생성
단어사전 생성
임베딩 행렬 생성
'''

# 6월 18일(수) - 오후 -
'''
NLP(자연어처리) -> RNN(순환신경망)
대상데이터 : 시계열데이터 < 시간,순서에 따라 뜻이 있는 데이터 >
원리 : 과거의 기억을 저장하여 현재 학습에 반영(연산)
SimpleRNN을 이용한 실습 - IMDB 영화리뷰 감성 분석 실습
-> tensorflow.datasets.imdb.load_data()
***
입력층(데이터) : 전처리 ( 토큰화, 단어 사전, 문장길이 균일화)

은닉층
EmbeddingLayer(output_dim) : 단어 임베딩
>> 가중치 행렬(참조 인덱스)(10000,32) : 단어사전과 같은 크기, 열은 output_dims로 정함.
>>> 가중치 행렬로 리뷰1개(문단)에 대한 임베딩 행렬 생성(470,32)
RNN layer(units) : 문장(문단) 임베딩
이렇게 만든 임베딩 행렬을 연산하여 결과물(1,16)이라는 일정한 숫자 배열 생성

출력층
Dense layer(units) : 6월19일에 계속..

절차
text data -> 단어 : 임베딩 벡터 -> 입력 문장(문장): 임베딩 벡터 -> 전체 text data에 대한 임베딩 행렬(특성 추출) -> 예측(분류)
'''

# 실습 IMDB 영화 리뷰 감성 분석(긍정 부정 분류)

## 데이터 다운로드
import tensorflow as tf

# 텍스트 데이터 다운로드
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb,load_data(num_words=10000)

# 데이터 분석 < 토큰화, 단어 사전 생성, 문장길이 균일화 >

# 학습용 데이터의 단어 수 추출

# 결과 값 저장할 리스트 생성
length_list = []

# for문 + len()함수
for review in X_train:
    length_list.append(len(review))

# 결과 확인
print(length_list)

# 통계 분석 -> 균일화 기준 90% 결정
import numpy as np
stats = np.percentile(a=length_list, q=[75,90,100])

# 데이터 전처리

### 학습용 / 평가용 리뷰의 길이를 일정하게 맞춰주기

# 최대 길이 설정
max_len = 470

# 자르고 붙이기(최신 토큰화 모델은 알아서 자르고 붙여줌)
X_train_pad = tf.keras.utils.pad_sequences(sequences=X_train, maxlen=max_len)
X_test_pad = tf.keras.utils.pad_sequences(sequences=X_test, maxlen=max_len)
