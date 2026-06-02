#include "mlir-vta/Target/VTABinaryEmitter.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::vta;

namespace {

static inline void setBits(uint64_t &word, unsigned lo, unsigned width,
                           uint64_t val) {
  uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
  assert((width == 64 || val <= mask) && "VTA field value exceeds bit width");
  word |= (val & mask) << lo;
}

// VTAMemInsn: LOAD(opcode=0) / STORE(opcode=1) / FINISH(opcode=3)
struct MemFields {
  uint64_t opcode = 0;
  bool pop_prev = false, pop_next = false, push_prev = false, push_next = false;
  uint64_t buffer_id = 0;
  uint64_t sram_base = 0, dram_base = 0;
  uint64_t y_size = 0, x_size = 0, x_stride = 0;
  uint64_t y_pad_top = 0, y_pad_bottom = 0, x_pad_left = 0, x_pad_right = 0;
};

static std::array<uint64_t, 2> packMem(const MemFields &f) {
  uint64_t w0 = 0, w1 = 0;
  setBits(w0, 0, 3, f.opcode);
  setBits(w0, 3, 1, f.pop_prev);
  setBits(w0, 4, 1, f.pop_next);
  setBits(w0, 5, 1, f.push_prev);
  setBits(w0, 6, 1, f.push_next);
  setBits(w0, 7, 3, f.buffer_id);
  setBits(w0, 10, 16, f.sram_base);
  setBits(w0, 26, 32, f.dram_base);
  setBits(w1, 0, 16, f.y_size);
  setBits(w1, 16, 16, f.x_size);
  setBits(w1, 32, 16, f.x_stride);
  setBits(w1, 48, 4, f.y_pad_top);
  setBits(w1, 52, 4, f.y_pad_bottom);
  setBits(w1, 56, 4, f.x_pad_left);
  setBits(w1, 60, 4, f.x_pad_right);
  return {w0, w1};
}

// Shared GEMM/ALU prefix on w0.
static void packLoopPrefix(uint64_t &w0, uint64_t opcode, bool pop_prev,
                           bool pop_next, bool push_prev, bool push_next,
                           bool reset, uint64_t uop_bgn, uint64_t uop_end,
                           uint64_t loop_out, uint64_t loop_in) {
  setBits(w0, 0, 3, opcode);
  setBits(w0, 3, 1, pop_prev);
  setBits(w0, 4, 1, pop_next);
  setBits(w0, 5, 1, push_prev);
  setBits(w0, 6, 1, push_next);
  setBits(w0, 7, 1, reset);
  setBits(w0, 8, 13, uop_bgn);
  setBits(w0, 21, 14, uop_end);
  setBits(w0, 35, 14, loop_out);
  setBits(w0, 49, 14, loop_in);
}

struct GemmFields {
  bool pop_prev = false, pop_next = false, push_prev = false, push_next = false;
  bool reset = false;
  uint64_t uop_bgn = 0, uop_end = 0, loop_out = 0, loop_in = 0;
  uint64_t dst_factor_out = 0, dst_factor_in = 0;
  uint64_t src_factor_out = 0, src_factor_in = 0;
  uint64_t wgt_factor_out = 0, wgt_factor_in = 0;
};

static std::array<uint64_t, 2> packGemm(const GemmFields &f) {
  uint64_t w0 = 0, w1 = 0;
  packLoopPrefix(w0, /*opcode=*/2, f.pop_prev, f.pop_next, f.push_prev,
                 f.push_next, f.reset, f.uop_bgn, f.uop_end, f.loop_out,
                 f.loop_in);
  setBits(w1, 0, 11, f.dst_factor_out);
  setBits(w1, 11, 11, f.dst_factor_in);
  setBits(w1, 22, 11, f.src_factor_out);
  setBits(w1, 33, 11, f.src_factor_in);
  setBits(w1, 44, 10, f.wgt_factor_out);
  setBits(w1, 54, 10, f.wgt_factor_in);
  return {w0, w1};
}

struct AluFields {
  bool pop_prev = false, pop_next = false, push_prev = false, push_next = false;
  bool reset = false;
  uint64_t uop_bgn = 0, uop_end = 0, loop_out = 0, loop_in = 0;
  uint64_t dst_factor_out = 0, dst_factor_in = 0;
  uint64_t src_factor_out = 0, src_factor_in = 0;
  uint64_t alu_opcode = 0;
  bool use_imm = false;
  uint64_t imm = 0;
};

static std::array<uint64_t, 2> packAlu(const AluFields &f) {
  uint64_t w0 = 0, w1 = 0;
  packLoopPrefix(w0, /*opcode=*/4, f.pop_prev, f.pop_next, f.push_prev,
                 f.push_next, f.reset, f.uop_bgn, f.uop_end, f.loop_out,
                 f.loop_in);
  setBits(w1, 0, 11, f.dst_factor_out);
  setBits(w1, 11, 11, f.dst_factor_in);
  setBits(w1, 22, 11, f.src_factor_out);
  setBits(w1, 33, 11, f.src_factor_in);
  setBits(w1, 44, 3, f.alu_opcode);
  setBits(w1, 47, 1, f.use_imm);
  setBits(w1, 48, 16, f.imm);
  return {w0, w1};
}

// VTAUop: dst_idx[0,11) src_idx[11,22) wgt_idx[22,32)
static uint32_t packUop(uint64_t dst, uint64_t src, uint64_t wgt) {
  uint64_t w = 0;
  setBits(w, 0, 11, dst);
  setBits(w, 11, 11, src);
  setBits(w, 22, 10, wgt);
  return static_cast<uint32_t>(w);
}

static void appendInsn(std::vector<uint8_t> &buf,
                       const std::array<uint64_t, 2> &words) {
  for (uint64_t w : words) {
    for (unsigned i = 0; i < 8; ++i)
      buf.push_back(static_cast<uint8_t>((w >> (8 * i)) & 0xFF));
  }
}

static void appendUop(std::vector<uint8_t> &buf, uint32_t u) {
  for (unsigned i = 0; i < 4; ++i)
    buf.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFF));
}

static LogicalResult writeFile(StringRef path, const std::vector<uint8_t> &buf) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec) {
    llvm::errs() << "vta-translate: cannot open " << path << ": "
                 << ec.message() << "\n";
    return failure();
  }
  os.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  os.flush();
  if (os.has_error()) {
    llvm::errs() << "vta-translate: error writing " << path << ": "
                 << os.error().message() << "\n";
    return failure();
  }
  return success();
}

} // namespace

LogicalResult mlir::vta::emitBinary(ModuleOp module, StringRef outDir) {
  std::vector<uint8_t> insnBuf;
  std::vector<uint8_t> uopBuf;

  for (Operation &op : module.getBody()->getOperations()) {
    if (auto u = dyn_cast<UopTableOp>(op)) {
      ArrayAttr dst = u.dst(), src = u.src(), wgt = u.wgt();
      if (dst.size() != src.size() || dst.size() != wgt.size()) {
        u.emitError("vta.uop_table dst/src/wgt arrays must have equal length (")
            << dst.size() << ", " << src.size() << ", " << wgt.size() << ")";
        return failure();
      }
      size_t n = dst.size();
      for (size_t i = 0; i < n; ++i) {
        uint64_t d = dst[i].cast<IntegerAttr>().getValue().getZExtValue();
        uint64_t s = src[i].cast<IntegerAttr>().getValue().getZExtValue();
        uint64_t w = wgt[i].cast<IntegerAttr>().getValue().getZExtValue();
        appendUop(uopBuf, packUop(d, s, w));
      }
    } else if (auto l = dyn_cast<LoadOp>(op)) {
      MemFields f;
      f.opcode = 0;
      f.pop_prev = l.pop_prev();
      f.pop_next = l.pop_next();
      f.push_prev = l.push_prev();
      f.push_next = l.push_next();
      f.buffer_id = static_cast<uint64_t>(l.buffer_id());
      f.sram_base = l.sram_base();
      f.dram_base = l.dram_base();
      f.y_size = l.y_size();
      f.x_size = l.x_size();
      f.x_stride = l.x_stride();
      f.y_pad_top = l.y_pad_top();
      f.y_pad_bottom = l.y_pad_bottom();
      f.x_pad_left = l.x_pad_left();
      f.x_pad_right = l.x_pad_right();
      appendInsn(insnBuf, packMem(f));
    } else if (auto s = dyn_cast<StoreOp>(op)) {
      MemFields f;
      f.opcode = 1;
      f.pop_prev = s.pop_prev();
      f.pop_next = s.pop_next();
      f.push_prev = s.push_prev();
      f.push_next = s.push_next();
      f.buffer_id = static_cast<uint64_t>(s.buffer_id());
      f.sram_base = s.sram_base();
      f.dram_base = s.dram_base();
      f.y_size = s.y_size();
      f.x_size = s.x_size();
      f.x_stride = s.x_stride();
      appendInsn(insnBuf, packMem(f));
    } else if (auto g = dyn_cast<GemmInsnOp>(op)) {
      GemmFields f;
      f.pop_prev = g.pop_prev();
      f.pop_next = g.pop_next();
      f.push_prev = g.push_prev();
      f.push_next = g.push_next();
      f.reset = g.reset();
      f.uop_bgn = g.uop_bgn();
      f.uop_end = g.uop_end();
      f.loop_out = g.loop_out();
      f.loop_in = g.loop_in();
      f.dst_factor_out = g.dst_factor_out();
      f.dst_factor_in = g.dst_factor_in();
      f.src_factor_out = g.src_factor_out();
      f.src_factor_in = g.src_factor_in();
      f.wgt_factor_out = g.wgt_factor_out();
      f.wgt_factor_in = g.wgt_factor_in();
      appendInsn(insnBuf, packGemm(f));
    } else if (auto a = dyn_cast<AluInsnOp>(op)) {
      AluFields f;
      f.pop_prev = a.pop_prev();
      f.pop_next = a.pop_next();
      f.push_prev = a.push_prev();
      f.push_next = a.push_next();
      f.reset = a.reset();
      f.uop_bgn = a.uop_bgn();
      f.uop_end = a.uop_end();
      f.loop_out = a.loop_out();
      f.loop_in = a.loop_in();
      f.dst_factor_out = a.dst_factor_out();
      f.dst_factor_in = a.dst_factor_in();
      f.src_factor_out = a.src_factor_out();
      f.src_factor_in = a.src_factor_in();
      f.alu_opcode = a.alu_opcode();
      f.use_imm = a.use_imm();
      f.imm = a.imm();
      appendInsn(insnBuf, packAlu(f));
    } else if (isa<FinishOp>(op)) {
      MemFields f;
      f.opcode = 3;
      appendInsn(insnBuf, packMem(f));
    } else if (op.hasTrait<OpTrait::IsTerminator>()) {
      // Module/region terminator: nothing to emit.
    } else if (op.getName().getDialectNamespace() == "vta") {
      // Unhandled (e.g. high-level) vta op: refuse to silently drop it.
      op.emitOpError("unhandled vta op in binary emitter; lower it before "
                     "translation");
      return failure();
    }
    // Non-vta, non-terminator ops are ignored.
  }

  std::string insnPath = (outDir + "/instructions.bin").str();
  std::string uopPath = (outDir + "/uop.bin").str();
  if (failed(writeFile(insnPath, insnBuf)))
    return failure();
  if (failed(writeFile(uopPath, uopBuf)))
    return failure();
  return success();
}
