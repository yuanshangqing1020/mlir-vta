# MLIR-VTA 文档索引

| 文档 | 说明 |
|------|------|
| [DESIGN_cn.md](DESIGN_cn.md) | 架构与设计总览 |
| [specs/phase1-gemm-design.md](specs/phase1-gemm-design.md) | 阶段一（16×16 GEMM 最小闭环）设计规格 |
| [specs/phase2-linalg-entry-design.md](specs/phase2-linalg-entry-design.md) | 阶段二（linalg 入口）设计规格 |
| [specs/phase3-generalized-gemm-design.md](specs/phase3-generalized-gemm-design.md) | 阶段三（通用维度 GEMM·多块）设计规格 |
| [plans/phase1-gemm.md](plans/phase1-gemm.md) | 阶段一实施计划 |
| [plans/phase2-linalg-entry.md](plans/phase2-linalg-entry.md) | 阶段二实施计划 |
| [plans/phase3-generalized-gemm.md](plans/phase3-generalized-gemm.md) | 阶段三实施计划（CASE1 + 多层） |
| [plans/phase3-overfit-strategy1.md](plans/phase3-overfit-strategy1.md) | 阶段三增量：Overfit strategy-1 多步调度（16×2064×16）✅ |
| [plans/phase3-overfit-strategy234.md](plans/phase3-overfit-strategy234.md) | 阶段三增量：Overfit strategy-2/3/4 全部落地 ✅ |
| [plans/phase3-multilayer-dram-allocation.md](plans/phase3-multilayer-dram-allocation.md) | 阶段三增量：多层 DRAM 地址分配（`-vta-dram-allocation`）✅ |
| [plans/phase3-alu-lowering.md](plans/phase3-alu-lowering.md) | 阶段三增量：ALU lowering（`vta.alu` + `--lower-vta-alu`，ADD_IMM 字节级验收）✅ |
| [plans/spike-bufferize-notes.md](plans/spike-bufferize-notes.md) | 阶段二 Task 0：tile + bufferize 管道 spike 记录 |
| [plans/spike-generalized-gemm-notes.md](plans/spike-generalized-gemm-notes.md) | 阶段三 Task 0：32×32 黄金解码 spike 记录 |

构建与命令行用法见仓库根目录 [README.md](../README.md)。
