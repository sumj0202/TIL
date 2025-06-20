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

-
# 6월 19일(목)

### 영화 리뷰 감성 분석 모델의 분석 실습

# Embedding(모델) - 은닉층
'''
매개변수 값 설정 -> 임베딩 생성 함수 호출, 모델 생성 -> 입력 데이터 생성 -> 출력 데이터 생성
'''
# 매개 변수의 값 설정
vocab_size =10000
embedding_size = 32
max_len = 470

# 임베딩 모델 호출 및 생성
embedding_layer = tf.keras.layers.Embedding(
    input_dim = vocab_size,
    output_dim = embedding_size,
    input_shape = (max_len,)
)

# 입력 데이터 생성
input = X_train_pad

# 입력 결과물 확인
out_embedding = embedding_layer(input)

#RNN(모델) - 은닉층
'''
생성 함수 호출, 모델 생성 -> 입력 결과물 확인
'''
rnn_layer = tf.keras.layers.SimpleRNN(
    units=16
)
# -> units은 (1,n)값으로 n의 수를 결정, 즉 임베딩 행렬에서 열의 값을 정함.

# 입력 결과물 확인
input = out_embedding
output_rnn = rnn_layer(input)

# Dense(모델) - 출력층 (입력의 총합 -> sigmoid함수 -> 확률로 변환)
'''
모델 생성 함수 호출 및 생성 -> 입력의 결과 확인 -> 예측값 저장
'''

# 모델 생성 및 호출
dense_layer = tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'
)

# 입력의 결과물 확인
input=output_rnn
output_dense=dense_layer(input)

# 정답 레이블로 변환
labels=[]
for output in output_dense:
    if output>=0.5:
        labels.append(1)
    else:
        labels.append(0)

# --> embedding layer, RNN layer , Dense layer는 변수가 연속적으로 연결(종속)적임
# --> 전 단계의 output이 그다음 단계의 input이 되는걸 기억

# 영화 리뷰 감성 분석 모델 생성
'''
모델 구성 순서
1.Ssequential()를 이용하여 모델 케이스 생성하기
2. Embedding layer 추가하기
3. RNN layer 추가하기
4. Dense layer 추가하기
'''

# 매개변수값 설정
vocab_size = 10000
embedding_size = 32
max_len = 470

# 기본 모델 생성
model = tf.keras.Sequential()

# Embedding layer 추가
model.add(tf.keras.layers.Embedding(
    input_dim = vocab_size,
    output_dim = embedding_size,
    input_shape = (max_len,)
))

# SimpleRNN layer 추가
model.add(tf.keras.SimpleRNN(units=16))

# Dense layer 추가
model.add(tf.keras.Dense(units=1, activation='sigmoid'))

# 생성된 모델 구조 확인
model.summary()

### 모델 compile
'''
손실 함수, 최적화 함수, 평가지표 설정
'''
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

### 모델 학습

'''
조기 종료 조건 설정 (early_stopping)
모델 저장 조건 설정 (checkpoint)
학습 진행 (fit)
'''

# 조기 종료 조건 설정
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights=True
)

# 모델 저장 조건 설정
file_path='저장경로/파일이름.keras'
checkpoint= tf.keras.callbacks.ModelCheckpoint(
    filepath=file_path,
    monitor='val_loss'
    save_best_only=True
)

# 학습 진행
model.fit(
    x=X_train_pad,
    y=y_train,
    batch_size=200,
    validation_split=0.2,
    epochs=10000000,
    callbacks=[early_stop,checkpoint]
)

# 모델 평가
'''
checkpoint로 인해 저장된 최적의 모델 불러오기 -> 모델 평가(evaluate)
'''

# 모델 불러오기
loaded_model=tf.keras.models.load_model(filepath=file_path)

# 모델 평가
result = loaded_model.evaluate(
    x=X_test_pad,
    y=y_test,
    batch_size=250
)

# LSTM모델은 오늘 배운 RNN layer를 lstm 으로 바꾼것뿐 나머지 똑같음 조금 단점을 개선한 정도

# 6월 20일 (수)

# 미국 기준 금리 예측 (LSTM 모델)
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
import tensorflow as tf
import datetime
import random

# FRED 데이터 다운로드를 위한 시작, 종료 시점 설정
start_date = datetime.datetime(2000,1,1)
end_date = datetime.datetime.now()

df_interest = pdr.DateReader('FEDFUNDS', 'fred', start_date, end_date)
print(df_interst)
# -> 350rows, 1col , non-null , 누락 없음

# 데이터 스케일링 모델 생성(전처리)
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df_interest)

# 전처리(2)
# 입력 데이터(시퀀스 데이터) 생성 함수 정의
def create_sequences(data, seq_length):
    '''
    ### 1. 함수의 기능 : 시계열 데이터를 LSTM 입력 형식의 시퀀스로 변환
    ### 2. 매개 변수 DATA : 스케일링된 시계열 데이터(2차원 배열)
    ### 3. 매개 변수 seq_length : 시간 간격(몇 개의 과거 데이터를 학습할 것인가?)
    ### 4. 결과 값 : X(입력 시퀀스), y(실제 금리)

    '''

    #결과 저장용 리스트 생성
    X_data = []
    y_data = []

    # 시간 간격 적용하여 입력 시퀀스 데이터 생성 (12개월)
    for i in range(len(data) - seq_length):
        X_data.append(data[i:i+seq_length,:])
        y_data.append(data[i+seq_length,:])

    return np.array(X_data), np.array(y_data)

# 시간 간격 설정
sequence_length = 12

# 함수 실행, 입력 데이터 생성

X_data, y_data = create_sequences(data=scaled_data, seq_length=sequence_length)

# 학습용 / 평가용 데이터 분할

'''
### 데이터 분할의 규칙 : 시간 순서 유지 (random X) --> 과거 데이터(학습용80%) + 최근 (평가용 20%)
'''

train_size=(int(len(X_data)*0.8))

X_train = X_data[:train_size,:,:]
X_test = X_data[train_size:,:,:]
y_train = y_data[:train_size,:,:]
y_test=y_data[train_size:,:,:]

# 모델 구성
'''
모델 구성 순서
1.Sequential()를 이용하여 모델 케이스 생성하기
2. LSTM layer 추가하기
3. Dense layer 추가하가
'''

# 모델 케이스 생성
model = tf.keras.Sequential()

# LSTM layer 추가
model.add(tf.keras.layers.LSTM(units=64,
                               return_sequences=True,
                               input_shape=(12,1)))

model.add(tf,keras.layers.LSTM(units=64))

### -> layer를 늘린다고 꼭 더 정교해지지는 않음.

# Dense layer 추가 --> 금리 1개 예측
model.add(tf.keras.layers.Dense(units=1))

# 모델 구조 확인
model.summary

'''
unit -> 뉴런 수라고 생각, unit이 클수록 더 다양하게 패턴을 학습, 대신 속도가 느려짐
그리고 너무많으면 overfitting 이라고, 오히려 성능 down

input_shape는 (12,1)인데 12는 학습의 주기라고 생각하면 되고, 1은 학습한 정보의 개수임.
즉, 12일간 하나의 금리를 input햇으니 12,1인거임. 30일 간격이었으면 30,1 인 거.

어제 배웠던 RNN과 똑같은데, 좀 더 성능이 개선된 거임.
'''