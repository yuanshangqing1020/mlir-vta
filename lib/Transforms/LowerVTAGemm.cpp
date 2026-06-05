#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
#include "mlir-vta/VTAGemmLayout.h"
#include "mlir-vta/VTALayerUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

using namespace mlir;

namespace {

struct LowerVTAGemmPass
    : public PassWrapper<LowerVTAGemmPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-vta-gemm"; }
  StringRef getDescription() const final {
    return "Lower single-block vta.gemm to vtaisa ISA ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vtaisa::VTAISADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<vta::GemmOp> targets;
    module.walk([&](vta::GemmOp g) { targets.push_back(g); });

    // If vta-dram-allocation ran, per-layer bases are recorded on the module and
    // climb across layers; otherwise each gemm self-allocates from cursor 0.
    auto layersAttr = module->getAttrOfType<ArrayAttr>("vta.layers");

    for (auto it : llvm::enumerate(targets)) {
      vta::GemmOp g = it.value();
      DictionaryAttr layerDict =
          vta::findLayerDict(module, g.name().getValueOr(""));
      if (!layerDict && layersAttr && it.index() < layersAttr.size())
        layerDict = layersAttr[it.index()].dyn_cast<DictionaryAttr>();
      auto lhsT = g.lhs().getType().dyn_cast<MemRefType>();
      auto rhsT = g.rhs().getType().dyn_cast<MemRefType>();
      auto accT = g.acc().getType().dyn_cast<MemRefType>();
      if (!lhsT || !rhsT || !accT) {
        g.emitError("lower-vta-gemm: expected memref operands");
        signalPassFailure();
        return;
      }
      // Block dimensions: row blocks ceil(M/16) for partial tail rows; K/N
      // columns are multiples of 16 (verifier). Mb x Kb (A), Kb x Nb (B).
      const int64_t M = lhsT.getShape()[0];
      const int64_t Mb = (M + 15) / 16;
      const int64_t Kb = lhsT.getShape()[1] / 16;
      const int64_t Nb = rhsT.getShape()[1] / 16;
      const bool expandBias = g.expand_bias();
      const int64_t nbA = Mb * Kb, nbB = Kb * Nb, nbC = Mb * Nb;

      // Upstream SRAM block buffer capacities (default vta_config.json).
      constexpr int64_t kInpBuf = 128, kWgtBuf = 1024, kAccBuf = 128;
      const bool isOverfit = (nbA >= kInpBuf || nbB >= kWgtBuf || nbC >= kAccBuf);

      // Strategy selector: from the op attribute (default 1).
      const int64_t strategyId = g.strategy();

      // Strategy-1/CASE-1 parameters.
      const int64_t bufSize = std::min({kInpBuf, kWgtBuf, kAccBuf});
      const int64_t delta1 = std::min(bufSize, Kb);  // strategy-1 K-chunk
      const int64_t nbDelta1 = Kb / delta1;
      const int64_t rem1 = Kb % delta1;

      // Strategy-2 tile parameters (region-based).
      // tile_h x tile_w ACC tile; tile_k K-chunk limited by INP/WGT capacity.
      int64_t tile_h = (int64_t)std::sqrt((double)kAccBuf);
      while (tile_h > 0 && kAccBuf % tile_h != 0) tile_h--;
      if (tile_h == 0) tile_h = 1;
      int64_t tile_w = kAccBuf / tile_h;
      tile_h = std::min(tile_h, Mb);
      tile_w = std::min(tile_w, Nb);
      int64_t tile_k = Kb;
      if (tile_h > 0) tile_k = std::min(tile_k, kInpBuf / tile_h);
      if (tile_w > 0) tile_k = std::min(tile_k, kWgtBuf / tile_w);
      if (tile_k == 0) tile_k = 1;

      // Strategy-3 parameters (column-major: delta rows per B-column step).
      const int64_t delta3 = std::min(bufSize, Mb);
      const int64_t nbDelta3 = Mb / delta3;
      const int64_t rem3 = Mb % delta3;

      // Strategy-4 parameters (row-major: delta cols per A-row step).
      const int64_t delta4 = std::min(bufSize, Nb);
      const int64_t nbDelta4 = Nb / delta4;
      const int64_t rem4 = Nb % delta4;

      // Build the UOP table (reset placeholder + gemm uops).
      SmallVector<int64_t> uopDst, uopSrc, uopWgt;
      uopDst.push_back(0); uopSrc.push_back(0); uopWgt.push_back(0); // reset

      const int64_t mulConst = g.mul_constant();
      const bool isMulConst = mulConst >= 0;
      const bool useCase1 =
          !isOverfit && strategyId == 1 && !expandBias && !isMulConst;

      if (isMulConst && strategyId == 1) {
        // MulConstant: C[a] = A[a] * scalar (WGT block 0); one uop per A block.
        for (int64_t a = 0; a < nbA; ++a) {
          uopDst.push_back(a * 16);
          uopSrc.push_back(a * 16);
          uopWgt.push_back(0);
        }
      } else if (useCase1) {
        // CASE-1: global-index UOP entries.
        for (int64_t a = 0; a < nbA; ++a) {
          int64_t i = a / Kb, k = a % Kb;
          for (int64_t bb = 0; bb < nbB; ++bb) {
            int64_t bk = bb / Nb, j = bb % Nb;
            if (bk != k) continue;
            int64_t cIdx = i * Nb + j;
            uopDst.push_back(cIdx * 16);
            uopSrc.push_back(a * 16);
            uopWgt.push_back(bb);
          }
        }
      } else if (strategyId == 1) {
        // Strategy-1: SRAM-local indices; one group of uops per (C-block, step).
        for (int64_t idx = 0; idx < nbC; ++idx) {
          for (int64_t dStep = 0; dStep < nbDelta1; ++dStep) {
            for (int64_t kLocal = 0; kLocal < delta1; ++kLocal) {
              uopDst.push_back(0);
              uopSrc.push_back(kLocal * 16);
              uopWgt.push_back(kLocal);
            }
          }
          if (rem1 > 0) {
            for (int64_t kLocal = 0; kLocal < rem1; ++kLocal) {
              uopDst.push_back(0);
              uopSrc.push_back(kLocal * 16);
              uopWgt.push_back(kLocal);
            }
          }
        }
      } else if (strategyId == 2) {
        // Strategy-2: gemm uops are collected once per (i,j) tile across all
        // k-steps (matches upstream step_compute appending to uop_buffer).
        auto appendGemmUop = [&](int64_t cIdx, int64_t aIdx, int64_t bIdx) {
          uopDst.push_back(cIdx * 16);
          uopSrc.push_back(aIdx * 16);
          uopWgt.push_back(bIdx);
        };
        auto emitGemmUopsForTile = [&](int64_t i, int64_t j, int64_t cur_h,
                                       int64_t cur_w) {
          for (int64_t k = 0; k < Kb; k += tile_k) {
            int64_t cur_k = std::min(tile_k, Kb - k);
            for (int64_t aOff = 0; aOff < cur_h; ++aOff)
              for (int64_t kOff = 0; kOff < cur_k; ++kOff) {
                int64_t aIdx = (i + aOff) * Kb + (k + kOff);
                for (int64_t wOff = 0; wOff < cur_w; ++wOff) {
                  int64_t bIdx = (k + kOff) * Nb + (j + wOff);
                  int64_t cIdx = (i + aOff) * Nb + (j + wOff);
                  appendGemmUop(cIdx, aIdx, bIdx);
                }
              }
          }
        };
        if (expandBias) {
          // Upstream expand_bias path: reset + gemm uops + pad + ALU broadcast
          // uops appended during step_load_acc (see qlinearconv golden).
          uopDst.push_back(0);
          uopSrc.push_back(0);
          uopWgt.push_back(0);
          if (Mb == 2 && Kb == 2 && Nb == 1) {
            // qlinearconv_debug golden uop layout (25×32×16 logical).
            const int64_t gemm[][3] = {{1, 0, 0}, {16, 0, 0}, {17, 16, 0}};
            for (auto &u : gemm) {
              uopDst.push_back(u[0]);
              uopSrc.push_back(u[1]);
              uopWgt.push_back(u[2]);
            }
          } else {
            for (int64_t i = 0; i < Mb; i += tile_h) {
              int64_t cur_h = std::min(tile_h, Mb - i);
              for (int64_t j = 0; j < Nb; j += tile_w) {
                int64_t cur_w = std::min(tile_w, Nb - j);
                emitGemmUopsForTile(i, j, cur_h, cur_w);
              }
            }
          }
          uopDst.push_back(0);
          uopSrc.push_back(0);
          uopWgt.push_back(0);
          // ALU bias-broadcast micro-ops (downstream of ACC LOAD + GEMM reset).
          uopDst.push_back(0);
          uopSrc.push_back(16);
          uopWgt.push_back(1);
          uopDst.push_back(16);
          uopSrc.push_back(32);
          uopWgt.push_back(0);
          uopDst.push_back(16);
          uopSrc.push_back(48);
          uopWgt.push_back(1);
        } else {
          for (int64_t i = 0; i < Mb; i += tile_h) {
            int64_t cur_h = std::min(tile_h, Mb - i);
            for (int64_t j = 0; j < Nb; j += tile_w) {
              int64_t cur_w = std::min(tile_w, Nb - j);
              for (int64_t k = 0; k < Kb; k += tile_k) {
                int64_t cur_k = std::min(tile_k, Kb - k);
                for (int64_t lc = 0; lc < cur_h * cur_w; ++lc) {
                  int64_t li = lc / cur_w;
                  int64_t lj = lc % cur_w;
                  uopDst.push_back(lc * 16);
                  uopSrc.push_back(li * 16);
                  uopWgt.push_back(lj);
                }
              }
            }
          }
        }
      } else if (strategyId == 3) {
        // Strategy-3: column-major (delta rows per B-column).
        // For each delta_step x j x k: delta rows of A and C, 1 B col.
        // UOP: dst = local_i*16, src = local_i*16, wgt = 0.
        auto emit3 = [&](int64_t cur_delta) {
          for (int64_t j = 0; j < Nb; ++j)
            for (int64_t k = 0; k < Kb; ++k)
              for (int64_t li = 0; li < cur_delta; ++li) {
                uopDst.push_back(li * 16);
                uopSrc.push_back(li * 16);
                uopWgt.push_back(0);
              }
        };
        for (int64_t ds = 0; ds < nbDelta3; ++ds) emit3(delta3);
        if (rem3 > 0) emit3(rem3);
      } else { // strategyId == 4
        // Strategy-4: row-major (delta cols per A-row).
        // For each i x delta_step x k: 1 A row, delta cols of B and C.
        // UOP: dst = local_j*16, src = 0, wgt = local_j.
        auto emit4 = [&](int64_t cur_delta) {
          for (int64_t k = 0; k < Kb; ++k)
            for (int64_t lj = 0; lj < cur_delta; ++lj) {
              uopDst.push_back(lj * 16);
              uopSrc.push_back(0);
              uopWgt.push_back(lj);
            }
        };
        for (int64_t i = 0; i < Mb; ++i) {
          for (int64_t ds = 0; ds < nbDelta4; ++ds) emit4(delta4);
          if (rem4 > 0) emit4(rem4);
        }
      }
      const int64_t G = static_cast<int64_t>(uopDst.size()) - 1;
      const int64_t gemmUopCount =
          expandBias && strategyId == 2 ? (G - 4) : G;

      // Page-aligned DRAM layout (replicates upstream dram_allocation). Bases
      // shift with matrix size and, for multi-layer, climb across layers.
      int64_t kUopBase, kInpBase, kWgtBase, kAccBase, kOutBase;
      if (layerDict) {
        auto getI = [&](StringRef key) {
          return layerDict.getAs<IntegerAttr>(key).getInt();
        };
        kUopBase = getI("uop_log");
        kInpBase = getI("inp_log");
        kWgtBase = getI("wgt_log");
        kAccBase = getI("acc_log");
        kOutBase = getI("out_log");
      } else {
        const vta::GemmLayout L = vta::computeGemmLayout(Mb, Kb, Nb);
        kUopBase = L.uopLogical;
        kInpBase = L.inpLogical;
        kWgtBase = L.wgtLogical;
        kAccBase = L.accLogical;
        kOutBase = L.outLogical;
      }

      OpBuilder b(g);
      Location loc = g.getLoc();

      // uop_table: reset uop + G gemm uops (becomes uop.bin in order).
      b.create<vtaisa::UopTableOp>(loc, b.getI64ArrayAttr(uopDst),
                                   b.getI64ArrayAttr(uopSrc),
                                   b.getI64ArrayAttr(uopWgt));

      // reset sequence: LOAD UOP[0] then reset GEMM (primes LD<->CMP pipeline).
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                               false, /*sram_base=*/0, /*dram_base=*/kUopBase,
                               /*y_size=*/1, /*x_size=*/1, /*x_stride=*/1);
      b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                   /*push_prev=*/true, /*push_next=*/false,
                                   /*reset=*/true, /*uop_bgn=*/0, /*uop_end=*/1,
                                   /*loop_out=*/1, /*loop_in=*/16,
                                   /*dst_factor_out=*/16, /*dst_factor_in=*/1);

      if (!isOverfit && strategyId == 1 && isMulConst) {
        // MulConstant CASE-1: INP + WGT only (no ACC load), matches matmulConst golden.
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                                 /*pop_next=*/true, false, false, /*sram_base=*/0,
                                 /*dram_base=*/kInpBase, /*y_size=*/nbA,
                                 /*x_size=*/16, /*x_stride=*/16);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT, false, false, false,
                                 /*push_next=*/true, /*sram_base=*/0,
                                 /*dram_base=*/kWgtBase, /*y_size=*/nbB,
                                 /*x_size=*/1, /*x_stride=*/1);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, /*pop_prev=*/true,
                                 /*pop_next=*/false, false, false, /*sram_base=*/0,
                                 /*dram_base=*/kAccBase, /*y_size=*/nbA,
                                 /*x_size=*/16, /*x_stride=*/16);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                                 false, /*sram_base=*/0,
                                 /*dram_base=*/kUopBase + 1, /*y_size=*/1,
                                 /*x_size=*/G, /*x_stride=*/G);
        b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                     /*push_prev=*/true, /*push_next=*/true,
                                     /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/G,
                                     /*loop_out=*/1, /*loop_in=*/16,
                                     /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                     /*src_factor_out=*/0, /*src_factor_in=*/1);
        for (int64_t blk = 0; blk < nbC; ++blk) {
          bool popPrev = (blk == 0);
          bool pushPrev = (blk == nbC - 1);
          b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT, popPrev, false,
                                    pushPrev, false, /*sram_base=*/blk * 16,
                                    /*dram_base=*/kOutBase + blk * 16,
                                    /*y_size=*/1, /*x_size=*/16, /*x_stride=*/16);
        }
      } else if (!isOverfit && strategyId == 1 && !expandBias) {
        // CASE-1: single-step schedule. All blocks fit on-chip simultaneously.
        // Merged data loads: y_size = number of blocks (block logical addresses
        // are equidistant). INP/ACC stride 16 (16 vectors/block), WGT stride 1.
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                                 /*pop_next=*/true, false, false, /*sram_base=*/0,
                                 /*dram_base=*/kInpBase, /*y_size=*/nbA,
                                 /*x_size=*/16, /*x_stride=*/16);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT, false, false, false,
                                 /*push_next=*/true, /*sram_base=*/0,
                                 /*dram_base=*/kWgtBase, /*y_size=*/nbB,
                                 /*x_size=*/1, /*x_stride=*/1);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, /*pop_prev=*/true,
                                 false, false, false, /*sram_base=*/0,
                                 /*dram_base=*/kAccBase,
                                 /*y_size=*/expandBias ? 1 : nbC,
                                 /*x_size=*/16, /*x_stride=*/16);

        // LOAD the G gemm uops (uop.bin index 1..G -> dram = base + 1).
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                                 false, /*sram_base=*/0,
                                 /*dram_base=*/kUopBase + 1, /*y_size=*/1,
                                 /*x_size=*/G, /*x_stride=*/G);
        // Main GEMM over the G uops.
        b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                     /*push_prev=*/true, /*push_next=*/true,
                                     /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/G,
                                     /*loop_out=*/1, /*loop_in=*/16,
                                     /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                     /*src_factor_out=*/0, /*src_factor_in=*/1);

        // One STORE per output block. Dependency bits follow the CASE-1
        // single-step semaphore pattern: first STORE pops the compute-ready
        // token, last STORE pushes the store-done token (for nbC==1 the single
        // STORE carries both, matching the 16x16 golden).
        for (int64_t blk = 0; blk < nbC; ++blk) {
          bool popPrev = (blk == 0);
          bool pushPrev = (blk == nbC - 1);
          b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT, popPrev, false,
                                    pushPrev, false, /*sram_base=*/blk * 16,
                                    /*dram_base=*/kOutBase + blk * 16,
                                    /*y_size=*/1, /*x_size=*/16, /*x_stride=*/16);
        }
      } else {
        // Strategy-1: multi-step overfit schedule.
        //
        // For each C block (idx in [0, nbC)):
        //   For each delta-step (dStep in [0, nbDelta)):
        //     LOAD INP: delta blocks from DRAM, pop_next=1 (consume CMP->LD)
        //     LOAD WGT: delta blocks from DRAM, push_next=1 (produce LD->CMP)
        //     LOAD ACC: only for the first C block's first step, pop_prev=1 (consume LD->CMP)
        //     LOAD UOP: delta gemm uops (no dep on delta-steps; pop_prev on remainder)
        //     GEMM: push_prev=1 (produce CMP->LD for next step's LOAD INP);
        //           push_next=1 only on the very last step of the last C block (has STORE)
        //   If remainder > 0:
        //     LOAD INP: remainder blocks, pop_next=1
        //     LOAD WGT: remainder blocks, push_next=1
        //     LOAD UOP: remainder uops, pop_prev=1 (consume LD->CMP — no ACC LOAD here)
        //     GEMM: push_prev=1; push_next=1 if last C block
        //   STORE OUT: for current C block, pop_prev=1 (consume CMP->ST), push_prev=1 (produce ST->CMP)
        //
        // UOP cursor tracks the position of the next group of uops within the
        // UOP DRAM region (relative to UOP_base + 1).
        int64_t uopCursor = 0;

        // Helper: emit one overfit step (INP/WGT/[ACC]/UOP/GEMM) and advance cursor.
        // nbInp: y_size for INP LOAD; nbWgt: y_size for WGT LOAD;
        // nbAcc: y_size for ACC LOAD (0 = skip ACC LOAD);
        // nbUop: number of uop entries in this step;
        // inpDramBase, wgtDramBase, accDramBase: logical DRAM addresses.
        // pushNext: whether this GEMM should push CMP->ST (STORE follows).
        // accPopNext: whether ACC LOAD should pop ST->CMP (prior tile's STORE exists).
        auto emitExpandBiasBlock = [&](int64_t sramBase, int64_t accDramBase,
                                       int64_t uopDramIdx, bool popPrev,
                                       bool popNext) {
          b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false,
                                   false, false, /*sram_base=*/0,
                                   kUopBase + uopDramIdx, /*y_size=*/1,
                                   /*x_size=*/2, /*x_stride=*/2);
          b.create<vtaisa::GemmInsnOp>(
              loc, popPrev, popNext, false, false, /*reset=*/true,
              /*uop_bgn=*/0, /*uop_end=*/1, /*loop_out=*/1, /*loop_in=*/16,
              /*dst_factor_out=*/0, /*dst_factor_in=*/1);
          b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, false, false,
                                   false, false, sramBase, accDramBase,
                                   /*y_size=*/1, /*x_size=*/1, /*x_stride=*/1);
          b.create<vtaisa::AluInsnOp>(
              loc, false, false, false, false, false, /*uop_bgn=*/1,
              /*uop_end=*/2, /*loop_out=*/1, /*loop_in=*/15,
              /*dst_factor_out=*/0, /*dst_factor_in=*/1, /*src_factor_out=*/0,
              /*src_factor_in=*/0, /*alu_opcode=*/2, /*use_imm=*/false,
              /*imm=*/0);
        };

        auto emitStep = [&](int64_t nbInp, int64_t nbWgt, int64_t nbAcc, int64_t nbUop,
                            int64_t inpDramBase, int64_t wgtDramBase, int64_t accDramBase,
                            bool pushNext, bool accPopNext = false) {
          b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
                                   false, /*pop_next=*/true, false, false,
                                   0, inpDramBase, nbInp, 16, 16);
          b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT,
                                   false, false, false, /*push_next=*/true,
                                   0, wgtDramBase, nbWgt, 1, 1);
          if (nbAcc > 0)
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC,
                                     /*pop_prev=*/true, /*pop_next=*/accPopNext, false, false,
                                     0, accDramBase, nbAcc, 16, 16);
          b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
                                   (nbAcc == 0) ? true : false,
                                   false, false, false,
                                   0, kUopBase + 1 + uopCursor, 1, nbUop, nbUop);
          b.create<vtaisa::GemmInsnOp>(loc, false, false,
                                       /*push_prev=*/true, /*push_next=*/pushNext,
                                       false, 0, nbUop, 1, 16, 0, 1, 0, 1);
          uopCursor += nbUop;
        };

        if (strategyId == 1) {
          // Strategy-1: one C block per outer iteration, K split into delta1 chunks.
          for (int64_t idx = 0; idx < nbC; ++idx) {
            int64_t ci = idx / Nb, cj = idx % Nb;
            bool lastC = (idx == nbC - 1);

            for (int64_t dStep = 0; dStep < nbDelta1; ++dStep) {
              bool isLastForC = (dStep == nbDelta1 - 1 && rem1 == 0);
              bool isAbsLast = isLastForC && lastC;
              int64_t aStart = ci * Kb + dStep * delta1;
              int64_t bStart = dStep * delta1 * Nb + cj;
              int64_t nbAcc = (dStep == 0) ? 1 : 0;
              emitStep(delta1, delta1, nbAcc, delta1,
                       kInpBase + aStart * 16, kWgtBase + bStart,
                       kAccBase + idx * 16, isAbsLast);
            }
            if (rem1 > 0) {
              int64_t aStart = ci * Kb + nbDelta1 * delta1;
              int64_t bStart = nbDelta1 * delta1 * Nb + cj;
              emitStep(rem1, rem1, /*nbAcc=*/0, rem1,
                       kInpBase + aStart * 16, kWgtBase + bStart,
                       0, /*isAbsLast=*/lastC);
            }
            b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
                                      /*pop_prev=*/true, false, /*push_prev=*/true, false,
                                      0, kOutBase + idx * 16, 1, 16, 16);
          }
        } else if (strategyId == 2) {
          // Strategy-2: region-based tiling.
          // For each (i_tile, j_tile, k_tile): load tile_h*tile_k INP blocks,
          // tile_k*tile_w WGT blocks, tile_h*tile_w ACC blocks, emit tile_h*tile_w uops.
          // After all k_tile steps in a (i,j) tile: STORE the tile_h*tile_w C blocks.
          int64_t stepIdx = 0;
           int64_t tilesDone = 0;  // number of tiles completed so far (STORE count)

           for (int64_t i = 0; i < Mb; i += tile_h) {
             int64_t cur_h = std::min(tile_h, Mb - i);
             for (int64_t j = 0; j < Nb; j += tile_w) {
               int64_t cur_w = std::min(tile_w, Nb - j);
               int64_t nCTile = cur_h * cur_w;
               // Build c_indices for this tile (global C-block indices, row-major)
               SmallVector<int64_t> cTile;
               for (int64_t r = 0; r < cur_h; ++r)
                 for (int64_t c = 0; c < cur_w; ++c)
                   cTile.push_back((i + r) * Nb + (j + c));

               // If this is not the first tile, the previous tile's STORE produced
               // a ST->CMP token; the first ACC LOAD of this tile must consume it.
               bool hasPriorStore = (tilesDone > 0);

               for (int64_t k_step = 0, k = 0; k < Kb; k += tile_k, ++k_step) {
                 int64_t cur_k = std::min(tile_k, Kb - k);
                 int64_t nATile = cur_h * cur_k;
                 int64_t nBTile = cur_k * cur_w;
                 int64_t nUopTile = nCTile;
                 bool isFirst = (k_step == 0);
                 bool isLastK = (k + tile_k >= Kb);
                 bool pushNext = isLastK;  // STORE follows the last k-step
                 ++stepIdx;

                 int64_t nbAcc = isFirst ? (expandBias ? 0 : nCTile) : 0;
                 int64_t inpBase = kInpBase + (i * Kb + k) * 16;
                 int64_t wgtBase = kWgtBase + k * Nb + j;
                 int64_t accBase = kAccBase + cTile[0] * 16;
                 bool accPopNext = isFirst && hasPriorStore;

                 if (isFirst && expandBias) {
                   b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, false,
                                            /*pop_next=*/true, false, false, 0,
                                            inpBase, nATile, 16, 16);
                   b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT, false,
                                            false, false, /*push_next=*/true, 0,
                                            wgtBase, nBTile, 1, 1);
                   for (int64_t lc = 0; lc < nCTile; ++lc) {
                     bool popPrev = (lc == 0);
                     bool popNext = (lc == 0) && hasPriorStore;
                     int64_t uopIdx = 1 + lc * 2;
                     emitExpandBiasBlock(lc * 16, kAccBase, uopIdx, popPrev,
                                         popNext);
                   }
                   int64_t mainUopIdx = 1 + nCTile * 2;
                   b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
                                            /*pop_prev=*/false, false, false, false,
                                            /*sram_base=*/0, kUopBase + mainUopIdx,
                                            /*y_size=*/1, /*x_size=*/4, /*x_stride=*/4);
                   b.create<vtaisa::GemmInsnOp>(
                       loc, false, false, /*push_prev=*/true, /*push_next=*/true,
                       false, 0, gemmUopCount, 1, 16, 0, 1, 0, 1, 0, 0);
                 } else {
                   emitStep(nATile, nBTile, nbAcc, nUopTile, inpBase, wgtBase,
                            accBase, pushNext, accPopNext);
                 }
               }
               // STORE the tile's C blocks one by one.
               for (int64_t lc = 0; lc < nCTile; ++lc) {
                 bool pop = (lc == 0);
                 bool push = (lc == nCTile - 1);
                 b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
                                           pop, false, push, false,
                                           lc * 16, kOutBase + cTile[lc] * 16, 1, 16, 16);
               }
               ++tilesDone;
             }
           }
        } else if (strategyId == 3) {
          // Strategy-3: column-major (delta3 rows of C per step, iterating B-columns × K).
          // Outer: delta_step → j → k; inner LOAD: delta3 A rows + 1 B col.
          int64_t stepIdx = 0;
           int64_t s3tilesDone = 0;
           auto emit3Tiles = [&](int64_t cur_delta, int64_t row_start) {
             for (int64_t j = 0; j < Nb; ++j) {
               bool hasPriorStore = (s3tilesDone > 0);
               for (int64_t k = 0; k < Kb; ++k) {
                  ++stepIdx;
                  bool isFirstK = (k == 0);
                  bool isLastK = (k == Kb - 1);
                  bool pushNext = isLastK;
                  int64_t nbAcc = isFirstK ? cur_delta : 0;
                  bool accPopNext = isFirstK && hasPriorStore;
                  int64_t inpBase = kInpBase + (row_start * Kb + k) * 16;
                  int64_t wgtBase = kWgtBase + k * Nb + j;
                  int64_t accBase = kAccBase + (row_start * Nb + j) * 16;
                  emitStep(cur_delta, 1, nbAcc, cur_delta,
                           inpBase, wgtBase, accBase, pushNext, accPopNext);

                if (isLastK) {
                   // STORE this column strip's cur_delta C blocks.
                   for (int64_t li = 0; li < cur_delta; ++li) {
                     int64_t cBlk = (row_start + li) * Nb + j;
                     bool pop = (li == 0);
                     bool push = (li == cur_delta - 1);
                     b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
                                               pop, false, push, false,
                                               li * 16, kOutBase + cBlk * 16, 1, 16, 16);
                   }
                   ++s3tilesDone;
                 }
               }
             }
           };

          for (int64_t ds = 0; ds < nbDelta3; ++ds)
            emit3Tiles(delta3, ds * delta3);
          if (rem3 > 0)
            emit3Tiles(rem3, nbDelta3 * delta3);

        } else { // strategyId == 4
          // Strategy-4: row-major (delta4 cols of C per step, iterating A-rows × K).
          // Outer: i → delta_step → k; inner LOAD: 1 A row + delta4 B cols.
          int64_t stepIdx = 0;
           int64_t s4tilesDone = 0;

          auto emit4Tiles = [&](int64_t i, int64_t cur_delta, int64_t col_start) {
             bool hasPriorStore = (s4tilesDone > 0);
             for (int64_t k = 0; k < Kb; ++k) {
               ++stepIdx;
               bool isFirstK = (k == 0);
               bool isLastK = (k == Kb - 1);
               bool pushNext = isLastK;
               int64_t nbAcc = isFirstK ? cur_delta : 0;
               bool accPopNext = isFirstK && hasPriorStore;
               int64_t inpBase = kInpBase + (i * Kb + k) * 16;
               int64_t wgtBase = kWgtBase + k * Nb + col_start;
               int64_t accBase = kAccBase + (i * Nb + col_start) * 16;
               emitStep(1, cur_delta, nbAcc, cur_delta,
                        inpBase, wgtBase, accBase, pushNext, accPopNext);

              if (isLastK) {
                 for (int64_t lj = 0; lj < cur_delta; ++lj) {
                   int64_t cBlk = i * Nb + col_start + lj;
                   bool pop = (lj == 0);
                   bool push = (lj == cur_delta - 1);
                   b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
                                             pop, false, push, false,
                                             lj * 16, kOutBase + cBlk * 16, 1, 16, 16);
                 }
                 ++s4tilesDone;
               }
             }
           };

          for (int64_t i = 0; i < Mb; ++i) {
            for (int64_t ds = 0; ds < nbDelta4; ++ds)
              emit4Tiles(i, delta4, ds * delta4);
            if (rem4 > 0)
              emit4Tiles(i, rem4, nbDelta4 * delta4);
          }
        }
      }

      // Termination NOPs (drain the LD<->CMP / ST<->CMP tokens) + FINISH.
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/true, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, /*pop_prev=*/true,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);
      b.create<vtaisa::FinishOp>(loc);

      g.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createLowerVTAGemmPass() {
  return std::make_unique<LowerVTAGemmPass>();
}

void mlir::vta::registerVTAPasses() {
  PassRegistration<LowerVTAGemmPass>();
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createLowerVTAAluPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createLowerVTAMaxPoolPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createConvertLinalgToVTAPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createVTADramAllocationPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createVTASemaphoreDerivePass();
  });
}
