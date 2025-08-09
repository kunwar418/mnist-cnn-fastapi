import streamlit as st
import requests
from PIL import Image
import io

st.title('mnist digit pred')
f=st.file_uploader('uplod img',type=['png','jpg','jpeg'])

if st.button('pred'):
    if f:
        img=Image.open(f)
        buf=io.BytesIO()
        img.save(buf,format='PNG')
        buf.seek(0)
        r=requests.post('http://127.0.0.1:8000/predict',files={'file':buf})
        if r.status_code==200:
            res=r.json()
            st.write('pred digit:',res['predicted_digit'])
            st.write('confidence:',res['confidence'])
