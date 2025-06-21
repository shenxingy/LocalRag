# -*- coding: utf-8 -*-
"""
一个基于 LangChain 和本地大语言模型（LLM）的检索增强生成 (RAG) 问答系统。

核心功能：
1.  从指定目录加载多种格式的文档（.txt, .pdf, .docx, .md）。
2.  赋予一个特殊的 'memo.txt' 文件最高优先级，作为回答问题的“权威备忘录”。
3.  使用 HuggingFace 模型进行文本嵌入，并存储在 Chroma 向量数据库中。
4.  利用 LangChain 表达式语言 (LCEL) 构建一个复杂的问答链，该链包含角色扮演、
    思维链 (CoT) 以及对权威备忘录的优先处理逻辑。
5.  通过命令行与用户交互，接收问题并基于提供的文档生成答案。
"""

# --- 基础库导入 ---
import os
import shutil
import argparse
from typing import List, Any, Dict
from operator import itemgetter

# --- LangChain 及社区库导入 ---
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# --- 全局配置 (Global Configuration) ---
DOCS_PATH = "docs"  # 普通文档所在目录
MEMO_FILE_PATH = "memo.txt"  # 权威备忘录文件路径
CHROMA_DB_PATH = "./chroma_db"  # Chroma 向量数据库持久化路径
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 使用的嵌入模型
OLLAMA_MODEL_NAME = "qwen2:7b"  # 使用的 Ollama 大语言模型
DEVICE = "cpu"  # 嵌入模型运行设备 ('cpu', 'cuda', etc.)


def load_and_split_documents(docs_path: str, memo_path: str) -> List[Any]:
    """
    加载并分割指定路径下的所有文档。

    特别处理 `memo.txt`，为其添加 `is_memo: True` 的元数据以示区分。
    """
    loader_mapping = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
    }

    all_documents = []
    print("--- 正在加载普通文档... ---")
    for ext, loader_class in loader_mapping.items():
        try:
            loader = DirectoryLoader(
                docs_path,
                glob=f"**/*{ext}",
                loader_cls=loader_class,
                show_progress=True,
                use_multithreading=True,
                # 避免加载备忘录文件（如果它在 docs 目录中）
                exclude=[os.path.join(docs_path, os.path.basename(memo_path))]
            )
            loaded_docs = loader.load()
            if loaded_docs:
                all_documents.extend(loaded_docs)
                print(f"成功从 '{ext}' 文件加载 {len(loaded_docs)} 个页面/片段。")
        except Exception as e:
            print(f"加载 '{ext}' 文件时出错: {e}")

    print(f"\n--- 正在加载权威备忘录 '{memo_path}'... ---")
    if os.path.exists(memo_path):
        try:
            memo_loader = TextLoader(memo_path, encoding='utf-8')
            memo_docs = memo_loader.load()
            for doc in memo_docs:
                doc.metadata['is_memo'] = True
                doc.metadata['source'] = os.path.basename(memo_path) # 确保源文件名正确
            all_documents.extend(memo_docs)
            print(f"成功加载权威备忘录 '{memo_path}'。")
        except Exception as e:
            print(f"加载权威备忘录 '{memo_path}' 时出错: {e}")
    else:
        print(f"警告: 未找到备忘录文件 '{memo_path}'，已跳过。")

    if not all_documents:
        print("错误: 未能加载任何文档。请检查 'docs' 目录和 'memo.txt' 文件是否存在。")
        return []

    print("\n--- 正在分割所有文档... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"所有文档已成功分割成 {len(chunks)} 个文本块。")
    return chunks


def setup_vector_store(chunks: List[Any], force_recreate: bool = False) -> Chroma:
    """
    创建或加载向量数据库。

    如果数据库已存在且未指定 `force_recreate`，则直接加载；否则，创建新的数据库。
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )

    if os.path.exists(CHROMA_DB_PATH) and not force_recreate:
        print(f"\n--- 正在从 '{CHROMA_DB_PATH}' 加载现有向量数据库... ---")
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        print("向量数据库加载成功。")
    else:
        print(f"\n--- 正在于 '{CHROMA_DB_PATH}' 创建新的向量数据库... ---")
        if force_recreate and os.path.exists(CHROMA_DB_PATH):
            print("检测到 'force_recreate' 参数，正在删除旧数据库...")
            shutil.rmtree(CHROMA_DB_PATH)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print("向量数据库创建并保存成功。")

    return vector_store


def format_docs_with_source(docs: List[Any]) -> str:
    """
    将检索到的文档块格式化为带有明确来源标识的单一字符串。

    这是为了帮助 LLM 区分权威备忘录和其他参考文件。
    """
    if not docs:
        return "知识库中未找到相关文档。"

    formatted_strings = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '未知来源')
        is_memo = doc.metadata.get('is_memo', False)

        header = f"--- 参考文件 {i+1} ---\n"
        if is_memo:
            header += f"[来源: 权威备忘录 ({source})]\n"
        else:
            header += f"[来源: 普通文档 ({source})]\n"

        content = doc.page_content
        formatted_strings.append(f"{header}{content}\n")

    return "\n".join(formatted_strings)


def create_qa_chain(vector_store: Chroma):
    """
    使用 LCEL 创建 RAG 问答链。

    该链融合了角色扮演、思维链 (CoT) 和对权威备忘录的优先处理逻辑。
    """
    print("\n--- 正在初始化问答链 (LCEL + 角色扮演 + CoT)... ---")
    llm = OllamaLLM(model=OLLAMA_MODEL_NAME, temperature=0.0)

    # 设置检索器，增加 k 值以提高召回备忘录的概率
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

    # 最终的提示模板：融合了角色扮演和零样本思维链 (Zero-Shot CoT) - v2
    prompt_template_str = """你是一位严谨、精确的 AI 信息核查员。你的任务是根据一个信息包（其中可能包含“权威备忘录”和“普通文档”）来回答用户的问题。

**核心原则:**
1.  **首要目标**: 准确回答用户的问题。你的回答必须完全基于下方“信息包”中提供的内容，禁止使用任何外部知识。
2.  **信息源的权威等级**:
    - `[来源: 权威备忘录]` 的内容是最高级别的真相，拥有最终决定权。
    - `[来源: 普通文档]` 的内容是次要参考信息。
3.  **处理规则**:
    - **直接回答**: 如果答案只存在于一个来源中（无论是备忘录还是普通文档），直接使用该信息进行回答。
    - **补充信息**: 如果备忘录提及了某个主题（例如 “项目Alpha”）但没有回答问题的具体细节（例如 预算），你**必须**从普通文档中寻找补充信息来回答，前提是该信息不与备忘录冲突。
    - **解决冲突**: 如果备忘录和普通文档就同一问题提供了相互矛盾的信息（例如 负责人），你**必须**只采用备忘录的说法，并忽略普通文档中的矛盾信息。
4.  **承认局限**: 如果信息包中的任何内容都不足以回答问题，请明确指出：“根据现有文档，我无法回答此问题。”

---
**信息包 (上下文):**
{context}
---

**用户的问题**: {question}

---
**核查与思考过程 (请在内部遵循此结构，不要在最终答案中展示):**
1.  **理解问题**: 用户想知道关于 `{question}` 的什么信息？
2.  **扫描所有信息源**:
    - 我在 `[来源: 权威备忘录]` 中找到了关于 `{question}` 的信息吗？是什么？
    - 我在 `[来源: 普通文档]` 中找到了关于 `{question}` 的信息吗？是什么？
3.  **应用处理规则进行裁定**:
    - **情况A (信息无冲突)**: 答案只在一个地方，或多个来源信息一致或互补。我将综合这些信息来构建答案。特别是，如果备忘录没提细节，但普通文档提了（比如预算），我就用普通文档的。
    - **情况B (信息有冲突)**: 备忘录和普通文档说法不一（比如负责人）。我将只采用备忘录的答案。
    - **情况C (信息不存在)**: 所有文档都找不到答案。我将声明无法回答。
4.  **构建最终答案**: 基于我的裁定，我现在为用户构建最终的、清晰简洁的答案。所有答案都必须是中文。

**最终核查报告 (AI 的回答):**
"""
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"]
    )

    # --- 优化后的 LCEL 链构建 ---
    # 这是一个并行的 Runnable 结构，它会同时执行字典中的所有分支。
    # 1. 'context' 分支: 接收输入（一个字典），取出 "question" 的值，
    #    将其传递给 retriever，然后将检索结果用 format_docs_with_source 函数格式化。
    # 2. 'question' 分支: 简单地从输入中传递 "question" 的值。
    # 这两个分支的结果会组合成一个新的字典 {context: "...", question: "..."}，
    # 这个字典的结构正好是 prompt 所需要的。
    setup_and_retrieval = RunnableParallel(
        {
            "context": itemgetter("question") | retriever | format_docs_with_source,
            "question": itemgetter("question"),
        }
    )

    # 完整 RAG 链
    # 输入 -> 并行检索与格式化 -> 填入提示模板 -> LLM 生成答案
    qa_chain = setup_and_retrieval | prompt | llm

    print("问答链准备就绪。")
    return qa_chain


def main():
    """主函数，负责启动整个 RAG 系统。"""
    parser = argparse.ArgumentParser(description="本地 RAG 问答系统 (备忘录优先, CoT 版本)")
    parser.add_argument('--recreate-db', action='store_true', help='强制重新创建向量数据库')
    args = parser.parse_args()

    chunks = load_and_split_documents(DOCS_PATH, MEMO_FILE_PATH)
    if not chunks:
        return

    vector_store = setup_vector_store(chunks, force_recreate=args.recreate_db)
    qa_chain = create_qa_chain(vector_store)

    print("\n--- 本地 RAG 系统已启动。输入 'exit' 或按 Ctrl+C 退出。 ---")
    print("-" * 70)

    try:
        while True:
            query = input("您的问题: ")
            if query.strip().lower() == 'exit':
                break
            if not query.strip():
                continue

            print("\nAI 正在思考...")
            # LCEL 链的调用方式非常简洁。
            # 我们传递一个包含问题（作为 'question' 键）的字典。
            final_answer = qa_chain.invoke({"question": query})

            print("\nAI 回答:")
            print(final_answer)

            # 为了调试和透明度，可以重新运行检索步骤以显示引用来源。
            print("\n--- 引用来源 (调试信息) ---")
            retriever = vector_store.as_retriever(search_kwargs={"k": 7})
            source_documents = retriever.invoke(query)
            if not source_documents:
                print("未检索到相关来源文档。")
            else:
                for i, doc in enumerate(source_documents):
                    metadata = doc.metadata
                    source_name = metadata.get('source', '未知文件')
                    is_memo_doc = metadata.get('is_memo', False)
                    memo_tag = " [权威备忘录]" if is_memo_doc else ""
                    page_number = metadata.get('page', 'N/A')
                    print(f"- 文档 {i+1}: {source_name}{memo_tag}, 页码: {page_number}")
            print("-" * 70)

    except (KeyboardInterrupt, EOFError):
        print("\n检测到退出信号。")
    except Exception as e:
        print(f"\n发生了一个意外错误: {e}")
    finally:
        print("RAG 系统已关闭。")


if __name__ == "__main__":
    main()