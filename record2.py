# 6월 23일 (월) - 오전 -
'''
저번 주 진행한 미국 기준 금리 예측 마무리
목차
==미국 기준 금리 예측==
데이터 다운로드
데이터 전처리
학습용, 평가용 데이터 생성
모델 컴파일 ( 저번주 진행 )
모델 학습 - 오전 부터 -
성능 예측 및 평가
예측 결과 시각화
미래 예측 ( 2025년 6월 1일 미국 금리)
'''

### 모델 학습

# 조기 종료 설정
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss'
    patience=3,
    restore_best_weights=True
)

# 모델 저장 조건 설정
file_path =''
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=file_path,
    monitor='val_loss'
    save_best_only=True
)

# 모델 학습 설정(과거 자료)
model.fit(
    x=X_train,
    y=y_train,
    epochs=1000,
    batch_size=24,
    validation_split=0.1,
    callbacks=[early_stop,checkpoint]
)

# 학습된 모델 예측(최근 데이터로)
pred_scaled = model.predict(X_test)

# 예측 결과 복원 (스케일링 하기 전)
pred_test=scaler.inverse_transform(pred_scaled)

# 평가용 정답 복원(최근데이터의 정답)
y_test_original=scaler.inverse_transform(y_test)

# RMSE 계산(최근 데이터에 대한 성능 테스트)
rmse=root_mean_squared_error(y_test_original, pred_test)

### 예측 결과 시각화

# 이미지 크기 재설정
plt.figure(figsize=(14,7))

# 날짜 인덱스 추출
test_dates=df_interest.index[train_size+sequence_length:]

# 그래프 생성
plt.plot(test_dates, y_test_original, label='Actual Rate', color='blue')
plt.plot(test_dates, pred_test, label='Predicted Rate', color='red', linestyle='--')
plt.title("Interest Rate Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Interest Rate (%)")
plt.legend()
plt.grid(True)
plt.show()

### 미래 예측
last_sequence = scaled_data[-sequence_length:, :]
print(last_sequence)

# LSTM 입력 형식에 맞게 reshape --> (1, sequence_length, 1)
input_data = last_sequence.reshape((1,sequence_length,1))

# 다음 시점 예측
next_prediction_scaled = model.predict(input_data)

# 예측 값 복원
next_prediction=scaler.inverse_transform(next_prediction_scaled)
