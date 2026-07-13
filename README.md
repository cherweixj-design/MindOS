# MindOS

MindOS 是一个用于学习 AI 应用开发的轻量级 RAG 知识问答系统。

它能够读取 Markdown 知识文件，通过 Embedding 和向量检索找到相关内容，再结合当前会话历史，由大语言模型生成回答。

## 已实现功能

- DeepSeek 对话模型
- Markdown 文档读取
- 按段落切分文本
- SiliconFlow Embedding
- 内存向量数据库
- 余弦相似度检索
- 相似度阈值过滤
- RAG 知识问答
- 当前会话短期记忆
- Retriever Debug 模式
- 基础异常处理

## 系统流程

### 建立知识库

```text
Markdown 文件
    ↓
MarkdownLoader
    ↓
ParagraphSplitter
    ↓
SiliconFlowEmbedding
    ↓
InMemoryVectorStore
```

### 用户问答

```text
用户问题
    ↓
Retriever
    ↓
Embedding
    ↓
VectorStore
    ↓
相关知识
    ↓
PromptBuilder
    ├── System Prompt
    ├── Knowledge
    ├── Memory
    └── Question
    ↓
DeepSeekLLM
    ↓
回答
```

## 核心模块

- `Indexer`：组织知识库建立流程
- `Retriever`：检索与问题相关的知识
- `PromptBuilder`：组合知识、历史和当前问题
- `Memory`：保存当前程序运行期间的对话
- `DeepSeekLLM`：调用大语言模型
- `MindOS`：编排完整问答流程
- `main.py`：创建并连接所有对象

## 项目结构

```text
MindOS/
├── knowledge/
│   └── employee.md
├── src/
│   ├── config/
│   │   └── settings.py
│   ├── llm/
│   │   ├── base.py
│   │   └── deepseek.py
│   ├── memory/
│   │   └── memory.py
│   ├── prompt/
│   │   ├── system.py
│   │   └── prompt_builder.py
│   ├── rag/
│   │   ├── base_embedding.py
│   │   ├── base_loader.py
│   │   ├── base_retriever.py
│   │   ├── base_splitter.py
│   │   ├── base_vector_store.py
│   │   ├── markdown_loader.py
│   │   ├── paragraph_splitter.py
│   │   ├── siliconflow_embedding.py
│   │   ├── in_memory_vector_store.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   └── mindos.py
├── .env.example
├── .gitignore
├── main.py
├── pyproject.toml
└── README.md
```

## 环境要求

- Python 3.13 或更高版本
- uv
- DeepSeek API Key
- SiliconFlow API Key

## 安装依赖

```bash
uv sync
```

## 配置环境变量

复制配置模板：

```bash
cp .env.example .env
```

然后在 `.env` 中填写真实 API Key。

请勿将 `.env` 上传到 Git 仓库。

## 运行

```bash
uv run python main.py
```

输入以下内容退出：

```text
exit
```

也可以按 `Control + C`。

## 使用示例

```text
You: 员工一年可以休几天？

MindOS: 员工每年享有15天年假。
```

```text
You: 公司的董事长是谁？

MindOS: 当前知识库中没有相关信息。
```

## 配置说明

### 相似度阈值

```env
MIN_RETRIEVAL_SCORE=0.6
```

只有相似度达到阈值的知识，才会交给大语言模型。

当前的 `0.6` 是根据示例知识库和测试问题校准的。知识库发生变化后，需要重新测试。

### Debug 模式

```env
DEBUG=true
```

开启后，终端会显示 Retriever 找到的 Chunk 和相似度分数。

普通使用时建议设置为：

```env
DEBUG=false
```

## 当前限制

- 向量数据只保存在内存中
- 每次启动程序都会重新建立索引
- Memory 只在当前程序运行期间有效
- 当前只读取一个 Markdown 知识文件
- 当前没有实现上下文问题改写
- 当前没有 Web Search 和 Tool Calling

## 后续计划

- 持久化向量数据库
- 多文件知识库
- 跨会话长期记忆
- Query Rewriting
- Web Search
- Tool Calling
- Agent 路由
- 使用 Codex 辅助开发 MindOS V2
