import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Function to load and split website data
def website_loader(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

# Function to set up the retrieval chain
def setup_retrieval_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.2, convert_system_message_to_human=True)

    # set to optimize RAM usage
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever()

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    # set chain type to optimize RAM usage
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                               condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                               combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
                                               )
    return qa

# Function for chatting with the website
def chat_with_website(qa, query):
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    if not query.strip():
        return "Please enter a question."
    answer = qa.invoke({"question": query})
    if isinstance(answer, dict) and 'answer' in answer:
        return answer['answer']
    else:
        return "Sorry, I couldn't find an answer to your question."

# Streamlit app
st.title("Chat with Website")

# User input for website URL
url = st.text_input("Enter the website URL:")

if st.button("Load Website Data"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        try:
            website_data = website_loader(url)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(website_data)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
            vectorstore = Chroma.from_documents(splits, embeddings)
            st.session_state.qa = setup_retrieval_chain(vectorstore)
            st.success("Website data loaded and processed!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Chat interface (only show if data is loaded)
if "qa" in st.session_state:
    st.write("You can now chat with the website.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question or request a summary..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chat_with_website(st.session_state.qa, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please enter a website URL and click 'Load Website Data' to begin.")
