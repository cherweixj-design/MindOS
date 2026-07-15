"""Manual check: SiliconFlow Embedding API."""

import sys

from src.config.settings import Settings
from src.rag.siliconflow_embedding import SiliconFlowEmbedding


def main() -> int:
    if not Settings.EMBEDDING_API_KEY:
        print("未配置 SILICONFLOW_API_KEY，无法执行真实 Embedding 检查。")
        return 1

    embedding = SiliconFlowEmbedding()
    texts = ["员工每年有15天年假。", "病假需要提交相关证明。"]
    vectors = embedding.embed(texts)

    assert len(vectors) == len(texts), "向量数量应与输入文本数量一致"
    assert len(vectors) > 0, "应至少返回一个向量"
    assert all(len(v) > 0 for v in vectors), "每个向量应非空"

    print(f"向量数量：{len(vectors)}")
    print(f"每个向量的维度：{len(vectors[0])}")
    print(f"第一个向量的前5个数字：{vectors[0][:5]}")

    print("\n✓ SiliconFlow Embedding 真实检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
