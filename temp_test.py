from ai_utils import process_docs, get_compression_retriever


def make_vector_db(input_folder, chunks_folder, vector_db_dir):
    process_docs(input_folder, chunks_folder, vector_db_dir)   
    return

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


if __name__ == "__main__":
    # make_vector_db('Folders', 'Chunks', 'VectorDB')
    compression_retriever = get_compression_retriever('VectorDB', 'Chunks')
    llm = ChatOpenAI(temperature=0.7, max_tokens=256, model_name="gpt-4o-mini", api_key="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")
    

    template = """
    <|system|>
    انتا مساعد ذكي تجيب على الاسئلة باللغة العربية و بشكل واضح بدون أي إضافات
    context: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """ 

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    qa_chain = (
        {"context": compression_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    print(compression_retriever)
    query = "ما هي القوانين التي تنظم البث الإذاعي والتلفزيوني في المملكة العربية السعودية؟"
    result = qa_chain.invoke(query)
    print(result)
