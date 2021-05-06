import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np

df = pd.read_csv('KODEX200')
df['일자'] = pd.to_datetime(df['일자'], format='%Y%m%d')
df = df.set_index('일자')
# 인덱스를 날짜로
df.drop(['Unnamed: 0','전일종가','수정주가구분','종목코드', '수정비율', '대업종구분', '소업종구분', '종목정보', '수정주가이벤트'],axis=1, inplace=True)
# 필요없는 부분 제거

train = df.iloc[360:]
test = df.iloc[0:360]
# 12개월간의 데이터를 테스트셋, 그 이전은 트레이닝셋

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)
# 데이터스케일링으로 0~1 값으로 변환

train_df = pd.DataFrame(train_sc, columns=['현재가', '거래량', '거래대금', '시가', '고가', '저가'], index=train.index)
test_df = pd.DataFrame(test_sc, columns=['현재가', '거래량', '거래대금', '시가', '고가', '저가'], index=test.index)
# 데이터프레임화

# 180일이전의 데이터를 갖고 오늘의 현재가, 거래량, 거래대급, 시가, 고가, 저가를 구하고 싶다
# 각각의 요소들이 서로에게 영향을 준다고 생각해서
# 그러나 각각을 전부 shift하고 트레이닝/테스트로 나누는 것이 의미있나?
# 우선 현재가만 구해보자
train_df.drop(['거래량', '거래대금', '시가', '고가', '저가'], axis=1, inplace=True)
test_df.drop(['거래량', '거래대금', '시가', '고가', '저가'], axis=1, inplace=True)

for x in range(1, 181):
    train_df['shift_%d' %x] = train_df['현재가'].shift(x)

    test_df['shift_%d' %x] = test_df['현재가'].shift(x)
# 180일간의 데이터로 내일의 주가를 구하고 싶기에, 각 날짜(인덱스)에 직전 180일의 주가를 입력

train_x = train_df.drop('현재가', axis=1).dropna()
train_y = train_df.dropna()[['현재가']]
# 트레이닝셋 데이터를 각각 학습을 위해 나누기, shift로 발생한 누락값들 삭제~제일 과거 180일치

train_x =train_x.values
train_y = train_y.values # ndarray로 타입변경 ~ 학습시 필요

print(train_x.shape, train_y.shape)
# 4229일 간의 데이터, 중직전 180일의 데이터를 통해서 다음날의 데이터를 구한다
# lstm 모델은 train 데이터들에 대해서 3차원의 데이터가 필요 ~ 밑 코드 수행
train_x_lstm = train_x.reshape(train_x.shape[0], 180, 1) # 총 데이터 수, 각 학습고려데이터 수, 결과데이터 수

#lstm 모델 만들기
from keras.layers import LSTM
from keras.models import Sequential  # 선형모델?
from keras.layers import Dense # 기본적으로 Dense 사용
import keras.backend as K # 백엔드 모듈 가져오기
from keras.callbacks import EarlyStopping # overfitting 방지용

K.clear_session() # 모델수행 후 데이터가 남아있는 오류때문에 필요?

model = Sequential() # 모델은 선형모델로
model.add(LSTM(18, input_shape=(180,1))) # 18개 층
model.add(Dense(1))  # output = 1
model.compile(loss='mse', optimizer='adam')
model.summary()

es = EarlyStopping(monitor='loss', patience=2, verbose=1)
# loss가 더 이상줄지않거나 모니터값이 개선이 3회이상 없을때 그만두기 ~ overfit 방지

model.fit(train_x_lstm, train_y, epochs=100, batch_size=100, verbose=1, callbacks=[es])
# epochs는 100, batch size를 100으로 두고 실행

#결과는 43번째 epoch에서 멈춤, loss값은 0.0064
# 검증을 위해서, test데이터셋도 전처리한뒤 적용해보자

test_x = test_df.drop('현재가', axis=1).dropna()
test_y = test_df.dropna()[['현재가']]
test_x =test_x.values
test_y = test_y.values
test_x_lstm = test_x.reshape(test_x.shape[0], 180, 1)
score = model.evaluate(test_x_lstm,test_y, batch_size=10)
print(score)
# 0.1 정도가 나온다


test_pred = model.predict(test_x_lstm, batch_size=10)
print(test_pred, test_y)
plt.scatter(test_pred, test_y)
plt.xlabel('Prediction')
plt.ylabel('Real')
plt.show()

# 기본적으로 예측치가 낮게 나옴 + 0.3~0.4 구간이 어마어마하게 난리남
# 예측치가 낮은건 최근 1년 사이 주가 상승률이 급격하게 높아져서 그런가? 반대로 그전엔 그저그럼 추가로 이전에 폭락때문에
# 0.3~04 가격치 문제는 그 사이 값이 적나?