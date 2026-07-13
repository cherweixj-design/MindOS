from src.rag.siliconflow_embedding import (
    SiliconFlowEmbedding,
)


embedding = SiliconFlowEmbedding()

vectors = embedding.embed(
    [
        "员工每年有15天年假。",
        "病假需要提交相关证明。",
    ]
)

print("向量数量：", len(vectors))
print("每个向量的维度：", len(vectors[0]))
print("第一个向量的前5个数字：", vectors[0][:5])