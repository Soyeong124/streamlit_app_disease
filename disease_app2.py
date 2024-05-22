import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Kkma
import streamlit.components.v1 as components
import os
import platform


# 로컬 CSS 파일을 로드하는 함수 정의
def local_css(file_name):
    with open(file_name, encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Streamlit 앱에 로컬 CSS 적용
local_css('css/style.css')

# konlpy 설치를 위한 JAVA_HOME 환경 변수 설정
# os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64' 

java_home = os.getenv('JAVA_HOME', '')
if not java_home or not os.path.exists(java_home):
    system = platform.system()
    if system == 'Windows':
            possible_paths = [
                r'C:\Program Files\Java\jdk-21'
            ]
    else: 
        possible_paths = [
                r'/usr/lib/jvm/java-11-openjdk-amd64'
            ]
    for path in possible_paths:
            if os.path.exists(path):
                java_home = path
                break

os.environ['JAVA_HOME'] = java_home 
            


# Kkma 인스턴스 초기화
kkma = Kkma()

# 전처리된 데이터 및 모델 로드
with open('processed_data.pkl', 'rb') as f:
    data, tokenized_symptoms, tfidf_matrix, vectorizer = pickle.load(f)

sys_data = pd.read_csv('symptom_description_no_dupl.csv')

# 사용자가 입력한 증상을 기반으로 질병을 예측하는 함수
def predict(input_value_raw):
    # 입력값을 콤마(,)로 분할하여 리스트로 변환
    input_values = input_value_raw.split(",")

    # 선택된 증상 정보와 ID를 저장할 리스트와 집합 초기화
    selected_symptoms_info = []

    # selected_symptoms_id_set 집합 초기화: 선택된 증상의 ID를 저장하여 중복 방지
    selected_symptoms_id_set = set()

    # 입력값을 형태소 분석하고, 각 증상과의 유사도를 계산하여 예측
    input_tokenized = [" ".join(kkma.morphs(input_value)) for input_value in input_values]
    for input_value, input_tok in zip(input_values, input_tokenized):
        for index, row in sys_data.iterrows():
            symptom = row['symptoms']
            if symptom in input_value:
                # symptoms이 완전히 일치하는 경우에 대해 처리
                if row['symptom_ids'] not in selected_symptoms_id_set:
                    selected_symptoms_info.append((symptom, 1.0, row['symptom_ids'], row['symptom_describe']))
                    selected_symptoms_id_set.add(row['symptom_ids'])
    
        # 타겟 문장의 TF-IDF 벡터화
        target_vector = vectorizer.transform([input_tok])
    
        # 증상과 타겟 문장 간의 유사도 계산
        similarities = cosine_similarity(target_vector, tfidf_matrix)
    
        # 유사도를 리스트로 변환
        similarity_list = similarities.flatten()
        
        # 유사도 임계값 설정
        similarity_threshold = 0.35

        # 유사도 기반으로 상위 10개의 증상 선택
        top_indices = similarity_list.argsort()[-10:][::-1]  # 상위 10개의 인덱스를 유사도 기준으로 내림차순 정렬
        for idx in top_indices:
            if similarity_list[idx] > similarity_threshold:  # 유사도가 0보다 큰 경우만 처리
                symptom = data.loc[idx, 'symptoms']
                if data.loc[idx, 'symptom_ids'] not in selected_symptoms_id_set:
                    selected_symptoms_info.append((symptom, similarity_list[idx], data.loc[idx, 'symptom_ids'], data.loc[idx, 'symptom_describe']))
                    selected_symptoms_id_set.add(data.loc[idx, 'symptom_ids'])

    # 최종적으로 selected_symptoms_info에는 중복되지 않은 증상 정보가 최대 10개까지 저장됩니다.
    response = []
    for i, (symptom, similarity, symptom_id, symptom_description) in enumerate(selected_symptoms_info, 1):
        response.append({
            "symptom": symptom,
            "symptom_id": symptom_id,
            "similarity": similarity,
            "symptom_description": symptom_description
        })

    # similarity 값 기준으로 내림차순 정렬
    response = sorted(response, key=lambda x: x['similarity'], reverse=True)

    if not response:
        return None
    return response

# 질병 데이터 로드
df = pd.read_pickle('disease_data_final.pkl')
df = df.fillna(" ")
df['total_symptoms'] = df['symptoms'].apply(lambda x: len(x.split(', ')))




# Streamlit 애플리케이션 시작
st.title('✨어디아파?')
st.write(f'증상을 설명해주세요 \n관련 있는 질병을 찾아드려요') 

# 사용자에게 증상 설명 입력 받기
input_value_raw = st.text_area('증상입력', placeholder=f"증상이 여러개면 콤마(,)로 구분해주세요 \n \n예시) 두통이 있어요, 기침이 나요", height=150)

# 확인 버튼을 클릭하면 이전 정보 초기화
if st.button('확인'):
    if input_value_raw:
        # 로딩 표시 추가
        with st.spinner('분석 중...'):
            # 예측 함수 호출 & 상태 초기화
            st.session_state['response'] = predict(input_value_raw)
            st.session_state['selected_symptoms'] = []
            st.session_state['checked_symptoms'] = []
            
            if not st.session_state['response']:
                st.error("일치하는 증상이 없어요. 다시 입력해주세요.")
    else:
        st.warning("증상 설명을 입력해주세요")



# 예측된 증상 표시
if 'response' in st.session_state and st.session_state['response']:
    response = st.session_state['response']
    high_confidence_symptoms = [item for item in response if item['similarity'] >= 0.9]
    low_confidence_symptoms = [item for item in response if item['similarity'] < 0.9]

    if high_confidence_symptoms or low_confidence_symptoms:
        if high_confidence_symptoms:
            st.write(f"당신의 증상은 아래와 같아요")
            st.write(f"<p style = 'font-size : 16px';>해당하는 증상을 체크하고   \n  아래 '질병 예측하기' 버튼을 클릭해주세요 </p>", unsafe_allow_html=True)
            for item in high_confidence_symptoms:
                if st.checkbox(item['symptom'], key=f"high_{item['symptom']}"):
                    if item['symptom'] not in st.session_state['selected_symptoms']:
                        st.session_state['selected_symptoms'].append(item['symptom'])
        
        if low_confidence_symptoms:
            st.write("다음 증상이 예상돼요:")
            for item in low_confidence_symptoms:
                if st.checkbox(item['symptom'], key=f"low_{item['symptom']}"):
                    if item['symptom'] not in st.session_state['selected_symptoms']:
                        st.session_state['selected_symptoms'].append(item['symptom'])

        if st.session_state['selected_symptoms']:
            if st.button('질병 예측하기'):
                user_symptom_list = st.session_state['selected_symptoms']
                
                # 각 질병마다 일치하는 증상 수 계산
                df['match_count'] = df['symptoms'].apply(lambda x: len(set(user_symptom_list) & set(x.split(', '))) if isinstance(x, str) else 0)
                
                # 일치하는 증상 수가 0인 경우 해당 내용을 표시하지 않음
                top_diseases = df[df['match_count'] > 0].sort_values(by=['match_count', 'total_symptoms'], ascending=[False, True]).head(5)
                st.markdown('<div id="scroll-target"></div>', unsafe_allow_html=True)  # 스크롤 대상 요소 추가

                components.html("""
                    <h3 id="scroll-target" style="text-align: center;">🧬관련 있는 질병이에요!</h3>
                    <script>
                        document.getElementById('scroll-target').scrollIntoView();
                    </script>
                """, height=30)


                for index, row in top_diseases.iterrows():
                    with st.expander(f"{row['disease_name']}    \n   📖 일치하는 증상 수: {row['match_count']}   \n   🩺 증상:   {row['symptoms']}"):
                         # 질병 이미지 표시
                        if row['disease_img']:
                            st.image(row['disease_img'], use_column_width=True)
                        if row['detailed_symptoms'].strip():  # 상세 증상이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>상세 증상</h3><p>{row['detailed_symptoms']}</p>", unsafe_allow_html=True)
                        if row['department'].strip():  # 진료과가 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>진료과</h3><p>{row['department']}</p>", unsafe_allow_html=True)
                        if row['related_diseases'].strip():  # 관련 질환이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>관련 질환</h3><p>{row['related_diseases']}</p>", unsafe_allow_html=True)
                        if row['synonyms'].strip():  # 동의어가 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>동의어</h3><p>{row['synonyms']}</p>", unsafe_allow_html=True)
                        if row['disease_course'].strip():  # 질병 경과가 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>질병 경과</h3><p>{row['disease_course']}</p>", unsafe_allow_html=True)
                        if row['disease_specific_diet'].strip():  # 특별 식이 요법이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>특별 식이 요법</h3><p>{row['disease_specific_diet']}</p>", unsafe_allow_html=True)
                        if row['diet_therapy'].strip():  # 식이 요법이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>식이 요법</h3><p>{row['diet_therapy']}</p>", unsafe_allow_html=True)
                        if row['recommended_food'].strip():  # 추천 음식이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>추천 음식</h3><p>{row['recommended_food']}</p>", unsafe_allow_html=True)
                        if row['caution_food'].strip():  # 주의 음식이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>주의 음식</h3><p>{row['caution_food']}</p>", unsafe_allow_html=True)
                        if row['other_notes'].strip():  # 기타 참고사항이 비어 있지 않은 경우에만 출력
                            st.write(f"<h3>기타 참고사항</h3><p>{row['other_notes']}</p>", unsafe_allow_html=True)




                # 예측 후 선택 증상 초기화
                st.session_state['selected_symptoms'] = []
                st.session_state['checked_symptoms'] = []


                 # 스크롤 이동 스크립트 추가
                st.markdown("""
                    <script>
                    document.getElementById('scroll-target').scrollIntoView({ behavior: 'smooth' });
                    </script>
                """, unsafe_allow_html=True)

    else:
        st.write("일치하는 증상이 없어요. 다시 입력해주세요.")

# 증상 정보 버튼 추가
st.markdown("""
    <button class="link-button" onclick="window.open('https://secret-map-dc8.notion.site/4a4f6187c52a4375a9aacc964340a6c6?v=f910f03a1c564805a7a120cda97bd370&pvs=4', '_blank');">
                원하는 증상이 안 나오나요?</button>
    """, unsafe_allow_html=True)
