from pykiwoom.kiwoom import *
kiwoom = Kiwoom()
# 로그인
kiwoom.CommConnect(block=True)
# block 파라미터에 True 지정시, 로그인 되기전까지 이후 코드 실행 안됨

Kodex = '069500'  # 미리 알 수도 있고, kiwoom에 존재하는 매서드로 코드 및 종목명 알아내기

# TR요청 단일, block_request 매서드로 요청 DataFrame으로 출력
info = kiwoom.block_request(
    'opt10001',  # 종목 기본정보
    종목코드='069500',
    output='KODEX 200',
    next=0)  # KOA Studio에 해당opt가 갖는 정보 및 input output 정보가 있음

# TR요청시, 데이터양이 많으면 끊겨서 데이터를 보내줌 ~ 연속조회 필요
df = []
data = kiwoom.block_request(
    'opt10081',
    종목코드='069500',
    기준일자='20210430',
    수정주가구분=1,
    output='KODEX200 주식일봉차트 조회',
    next=0)
df.append(data)  # 단일조회한 것 먼저 df에 저장

while kiwoom.tr_remained:  # 불러오지 못하고 남아있는 것들에 대해서
    data = kiwoom.block_request(
        'opt10081',
        종목코드='069500',
        기준일자='20210430',
        수정주가구분=1,
        output='KODEX200 주식일봉차트 조회',
        next=2)
    df.append(data)
    time.sleep(2)  # 쉴 필요 있음

data = pd.concat(df)  # data데이터프레임에 추가로 받은 df데이터들 밑에 추가
data.to_csv('KODEX200')