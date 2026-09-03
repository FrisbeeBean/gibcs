import os
import streamlit as st
from inference import loader
@st.cache_resource
def ld_m():
    """Caches model and tokenizer loading to prevent reloading"""
    rd=os.path.dirname(os.path.abspath(__file__))
    tp=os.path.join(rd,"..","tokenizer","tokenizer.model")
    cp="gibcs_chckpt.pt"
    return loader(cp,tp)
def main():
    """Runs Streamlit UI for inference"""
    st.title("Model Inference")
    b=ld_m()
    pmpt=st.text_area("User >>",height=200)
    if st.button("Generate"):
        if pmpt.strip():
            with st.spinner("Generating..."):
                msg=b.gen(pmpt,mt=150,tmp=0.4,tk=40)
                st.write("Model >>")
                st.write(msg)
        else:
            st.warning("Please enter a prompt.")
if __name__=="__main__":
    main()