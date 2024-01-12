import streamlit as st
from PIL import Image # 이미지 입출력 담당하는 파이썬 표준 라이브러리
import requests

# Streamlit 페이지 설정
st.title('Face Mastk Detection')
st.write('Upload an image to classify it as with_mask or without_mask.')

# 이미지 업로드 위젯
uploaded_file = st.file_uploader("이미지를 선택하세요!", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  # 이미지 표시
  image = Image.open(uploaded_file)
  st.image(image, caption='Upload Image',
  use_column_width=True)
  st.write("구분하는 중...")
  server_url = "http://localhost:8000/resnet/predict" # FastAPI 서버 url
  files = {"file": uploaded_file.getvalue()}
  response = requests.post(server_url, files=files)

  # 결과 표시
  if response.status_code == 200:
    result = response.json()
    st.write(f'Prediction: {result["predicted_class"]}')
  else:
    st.write("Error in prediction")
