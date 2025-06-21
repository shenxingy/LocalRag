# 本地知识库问答系统 (RAG) - 备忘录优先与思维链实现

这是一个基于 LangChain 和本地大语言模型 (LLM) 构建的高级检索增强生成 (RAG) 问答系统。它不仅能从本地文档中检索信息回答问题，还实现了一套复杂的逻辑，使其能够优先采纳“权威备忘录”中的信息，并利用“思维链 (Chain-of-Thought)”技术来处理信息冲突和补充，从而提供更准确、更可靠的答案。

## 核心特性

  - **多格式文档支持**: 能自动加载并处理 `.txt`, `.pdf`, `.docx`, 和 `.md` 等多种常见格式的文档。
  - **“权威备忘录”优先**: `memo.txt` 文件被视为最高权威信息源。当其内容与其他文档冲突时，系统会优先采信备忘录。
  - **完全本地化运行**: 依赖本地大语言模型（通过 [Ollama](https://ollama.com/)）和本地嵌入模型，确保数据隐私和离线可用性。
  - **高性能向量存储**: 使用 [ChromaDB](https://www.trychroma.com/) 作为向量数据库，实现高效的语义检索和本地持久化。
  - **高级提示工程**:
      - **角色扮演 (Role-Playing)**: LLM 被赋予“信息核查员”的角色，使其行为更加严谨。
      - **思维链 (Chain-of-Thought, CoT)**: 提示词内包含了详细的思考步骤和处理规则，引导 LLM 如何分析、裁定和整合来自不同来源的信息。
  - **优雅的 LCEL 构建**: 利用 LangChain 表达式语言 (LCEL) 以声明式、可组合的方式构建了整个 RAG 链，代码清晰且易于维护。
  - **命令行交互**: 提供一个简单易用的命令行界面与系统进行问答。
  - **智能来源引用**: 在回答后，会列出用于生成答案的源文档，便于用户追溯和核实。

## 系统架构与工作流程

系统的工作流程遵循一个标准的 RAG 模式，但加入了特殊处理逻辑：

1.  **数据加载与预处理**:

      - 系统启动时，会扫描 `docs/` 目录下的所有支持的文档和根目录下的 `memo.txt`。
      - `memo.txt` 在加载时会被添加一个特殊的元数据标记 `is_memo: True`。
      - 所有文档被分割成大小适中的文本块 (Chunks)。

2.  **向量化与索引**:

      - 使用 `BAAI/bge-large-zh-v1.5` 模型将所有文本块转换为向量。
      - 这些向量被存储在 Chroma 向量数据库中。如果数据库已存在，则直接加载以节省时间。

3.  **用户提问与检索**:

      - 用户通过命令行输入问题。
      - 系统使用相同的嵌入模型将问题向量化。
      - 在 Chroma 数据库中进行语义相似度搜索，检索出与问题最相关的 K 个文本块。

4.  **上下文构建与提示**:

      - 检索到的文本块被格式化成一个“信息包”。每个文本块都会明确标出其来源是 **`[权威备忘录]`** 还是 **`[普通文档]`**。
      - 这个信息包和用户原始问题被一同填入一个精心设计的提示模板中。

5.  **LLM 推理与生成**:

      - 携带了角色、思考规则和上下文的完整提示被发送给本地 `qwen2:7b` 模型。
      - LLM 遵循提示中的指令：分析信息、解决冲突（优先采纳备忘录）、整合互补信息，并生成最终答案。

6.  **返回答案**:

      - LLM 生成的答案被展示给用户，同时附上本次回答引用的源文档列表作为参考。

## 技术栈

  - **核心框架**: [LangChain](https://www.langchain.com/)
  - **本地 LLM 服务**: [Ollama](https://ollama.com/) (模型: `qwen2:7b`)
  - **文本嵌入模型**: [HuggingFace BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
  - **向量数据库**: [ChromaDB](https://www.trychroma.com/)
  - **文档加载器**: `PyPDFLoader`, `Docx2txtLoader`, `TextLoader`

## 安装与配置

### 1\. 先决条件

  - **Python 3.8+**
  - **Ollama**: 请确保您已在本地安装并运行了 Ollama。
      - [下载并安装 Ollama](https://ollama.com/)
      - 在终端中拉取本项目所需的模型：
        ```bash
        ollama pull qwen2:7b
        ```

### 2\. 克隆项目与安装依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd <your-repo-folder>

# (推荐) 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# 安装所需的 Python 库
pip install -r requirements.txt
```

### 3\. 准备知识库文件

1.  在项目根目录下创建一个名为 `docs` 的文件夹。
2.  将您的普通参考文档（.pdf, .docx, .txt, .md）放入 `docs` 文件夹内。
3.  在项目根目录下创建一个名为 `memo.txt` 的文件。将最重要、最权威的信息写入此文件。

您的文件结构应如下所示：

```
.
├── your_script_name.py     # 主程序脚本
├── docs/                   # 普通文档目录
│   ├── project_plan.pdf
│   └── meeting_notes.docx
├── memo.txt                # 权威备忘录文件
└── requirements.txt        # Python 依赖
```

## 如何运行

通过终端在项目根目录下运行主脚本：

```bash
python your_script_name.py
```

**可选参数**:

  - `--recreate-db`: 如果您想强制删除旧的向量数据库并根据当前文档重新创建一个新的，请使用此参数。这在您更新了大量文档后非常有用。
    ```bash
    python your_script_name.py --recreate-db
    ```

系统启动后，会显示加载和处理文档的日志。当看到 `本地 RAG 系统已启动` 的提示后，您就可以开始提问了。

## 工作原理解析

### “权威备忘录”优先策略的实现

本项目的核心创新点在于如何让 LLM 理解并执行“备忘录优先”的规则。这并非通过修改检索算法，而是通过巧妙的 **元数据注入** 和 **提示工程** 实现的：

1.  **元数据注入**: 在 `load_and_split_documents` 函数中，当加载 `memo.txt` 时，会为它的每一个文本块添加 `{'is_memo': True}` 的元数据。
2.  **动态上下文格式化**: 在 `format_docs_with_source` 函数中，当检索到的文档块被送往 LLM 之前，该函数会检查每个文档块的元数据。如果 `is_memo` 为 `True`，它会在文本块前加上 `[来源: 权威备忘录]` 的标签，否则加上 `[来源: 普通文档]`。
3.  **在提示中明确规则**: 最终的提示词（`prompt_template_str`）中包含了非常明确的指令，告诉 LLM 这两种标签的权威等级差异以及如何处理它们之间的信息冲突与互补。

这种方法将复杂的业务逻辑转化为 LLM 可以直接理解的自然语言指令，充分利用了现代 LLM 强大的指令遵循能力。

### LCEL (LangChain Expression Language)

本项目使用 LCEL 来构建问答链，其优点是高度声明化和可组合性。`create_qa_chain` 函数中的这段代码是 LCEL 的精髓：

```python
setup_and_retrieval = RunnableParallel(
    {
        "context": itemgetter("question") | retriever | format_docs_with_source,
        "question": itemgetter("question"),
    }
)
qa_chain = setup_and_retrieval | prompt | llm
```

  - `|` (管道符): 将一个组件的输出作为下一个组件的输入，像 Linux 的管道一样连接操作。
