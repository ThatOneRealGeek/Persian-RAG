#کتابخانه های استفاده شده
import streamlit as st  # برای ساخت رابط کاربری وب
import pandas as pd     # برای ساخت و نمایش جدول‌ها
from langchain_community.document_loaders import WebBaseLoader  # برای خواندن متن از وب‌سایت‌ها
from langchain_text_splitters import RecursiveCharacterTextSplitter # برای تکه‌تکه کردن متن‌های طولانی
from langchain_huggingface import HuggingFaceEmbeddings # برای تبدیل متن به عدد (بردار)
from langchain_community.vectorstores import Chroma # دیتابیس برداری برای ذخیره متن‌ها
from langchain_community.llms import Ollama # برای ارتباط با مدل هوش مصنوعی Llama3
import os       # برای کار با فایل‌های سیستم عامل
import shutil   #پاک کردن دیتابیس قدیمی

#تنظیمات صفحه اصلی
st.set_page_config(page_title="دستیار فیلم", layout="wide", page_icon="🎬")

#CSS
st.markdown("""
<style>
    /* تنظیمات کلی صفحه */
    .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Vazirmatn', sans-serif;
    }
    
    /* تیتر وسط‌چین */
    .main-header {
        text-align: center; 
        color: #ff4b4b;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px #000000;
    }
    
    /* باکس جواب */
    .answer-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-right: 5px solid #ff4b4b;
        margin-top: 20px;
    }

    /* --- تنظیمات اجباری برای سایدبار و ورودی‌ها --- */
    
    /* راست‌چین کردن متن داخل ورودی‌ها و Text Area */
    .stTextInput input, .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }
    
    /* راست‌چین کردن لیبل (تیتر) بالای ورودی‌ها */
    .stTextArea label, .stTextInput label {
        width: 100%;
        text-align: right !important;
        display: flex;
        justify-content: flex-end; /* هل دادن متن به سمت راست */
    }
    
    /* راست‌چین کردن کل سایدبار */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    /* --- تنظیمات اجباری برای جدول‌ها --- */
    div[data-testid="stTable"] table {
        direction: rtl;
        width: 100%;
    }
    /* راست‌چین کردن تیتر ستون‌ها */
    div[data-testid="stTable"] th {
        text-align: right !important;
        direction: rtl;
    }
    /* راست‌چین کردن محتوای سلول‌ها */
    div[data-testid="stTable"] td {
        text-align: right !important;
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

#تحلیل و امبدینگ لینک ها
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PERSIST_DIRECTORY = "./chroma_db"

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def process_websites(urls):
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except:
            pass

    with st.spinner(' در حال مطالعه...'):
        loader = WebBaseLoader(urls)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)
        
        embedding_model = load_embedding_model()
        vector_db = Chroma.from_documents(
            documents=all_splits, 
            embedding=embedding_model, 
            persist_directory=PERSIST_DIRECTORY
        )
        return vector_db, len(all_splits)

def get_rag_response(query, vector_db):
    llm = Ollama(model="llama3")
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    relevant_docs = retriever.invoke(query)
    return relevant_docs, llm

#رابط کاربری
st.markdown('<h1 class="main-header">🎬 دستیار هوشمند نقد و بررسی فیلم</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🌐 منابع اطلاعاتی")
    input_urls = st.text_area(
        "لینک‌های وب‌سایت‌ها (هر خط یک لینک)", 
        value="https://fa.wikipedia.org/wiki/پدرخوانده\nhttps://fa.wikipedia.org/wiki/شوالیه_تاریکی_(فیلم)",
        height=150
    )
    urls = [url.strip() for url in input_urls.split('\n') if url.strip()]
    
    if st.button("🚀 پردازش منابع", use_container_width=True):
        if urls:
            try:
                vector_db, count = process_websites(urls)
                st.session_state['db_ready'] = True
                st.success(f"✅ {count} بخش از متن فیلم‌ها ذخیره شد.")
            except Exception as e:
                st.error(f"خطا: {e}")
        else:
            st.warning("لطفاً لینک وارد کنید.")

if st.session_state.get('db_ready'):
    embedding_model = load_embedding_model()
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model)
    
    query = st.text_input("🍿 سوال سینمایی خود را بپرسید:", placeholder="مثلاً: موضوع اصلی فیلم پدرخوانده چیست؟")
    
    if query:
        docs, llm = get_rag_response(query, vector_db)
        
        with st.spinner("🤖 هوش مصنوعی در حال نوشتن نقد..."):
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
#پرامت هوش مصنوعی
            prompt = f"""
            You are a helpful AI assistant that speaks ONLY Persian (Farsi).
            
            CRITICAL INSTRUCTIONS:
            1. Answer the user's question strictly in PERSIAN language.
            2. Do NOT write any English sentences.
            3. Start your answer directly in Persian text.
            4. Use the provided Context to answer.
            
            Context:
            {context_text}
            
            User Question: {query}
            """
            
            response = llm.invoke(prompt)
            
#نحوه نمایش جواب
            st.markdown(f"""
            <div class="answer-box" style="direction: rtl; text-align: right;">
                <h3 style="margin-bottom: 15px;">💡 پاسخ هوش مصنوعی:</h3>
                <div dir="auto" style="font-size: 1.1em; line-height: 1.8; text-align: start;">
                    {response}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # جدول منابع
        with st.expander("📚 مشاهده متن‌های پیدا شده در منابع (برای بررسی دقیق‌تر)"):
            table_data = []
            for doc in docs:
                table_data.append({
                    "منبع": doc.metadata.get('source', 'نامشخص'),
                    "بخشی از متن": doc.page_content[:300] + "...",
                })
                
                st.table(pd.DataFrame(table_data))

else:
    st.info("👈 برای شروع، لینک‌ها را در منوی سمت راست وارد کرده و دکمه پردازش را بزنید.")