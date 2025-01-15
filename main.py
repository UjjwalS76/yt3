import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Perplexity
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Function to load and split website data
def website_loader(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

# Function to set up the retrieval chain
def setup_retrieval_chain(vectorstore):
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    # llm = ChatGoogleGenerativeAI(model="gemini-pro")
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")
    llm = Perplexity(model="llama-3-sonar-small-128k-online", perplexity_api_key=st.secrets["PPLX_API_KEY"])

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
    # answer = qa({"question": query})
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    # Check for empty query
    if not query.strip():
        return "Please enter a question."
    answer = qa.invoke({"question": query})
    # Check if the answer is a dictionary and extract 'answer' key
    if isinstance(answer, dict) and 'answer' in answer:
        return answer['answer']
    else:
        # Handle cases where the expected key is not found or answer is not a dictionary
        return "Sorry, I couldn't find an answer to your question."

    return answer['answer']


# Streamlit app
st.title("Chat with Website")

# Website URL input
default_url = "https://www.vedabase.io/"
url = st.text_input("Enter website URL:", value=default_url)

if st.button("Load Website Data"):
    website_data = website_loader(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(website_data)

    # Use HuggingFace embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(splits, embeddings)
    st.session_state.qa = setup_retrieval_chain(vectorstore)  # Store qa in session state
    st.success("Website data loaded and processed!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "qa" in st.session_state:
    if prompt := st.chat_input("Ask a question about the website..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = chat_with_website(st.session_state.qa, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.write("Please load the website data first.")
