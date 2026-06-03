// VTASemaphoreDerivePass: re-derive dep bits using the 4-counter semaphore.
//
// Processes each "program" delimited by vtaisa.finish ops.  For each program,
// it walks the vtaisa ops, classifies each by role (INP/WGT/ACC load, UOP
// load, compute GEMM/ALU, STORE, NOP-drain, finish) and applies the upstream
// 4-counter state machine (CMP->LD / LD->CMP / CMP->ST / ST->CMP) to derive
// correct pop_prev/pop_next/push_prev/push_next bits.
//
// Because MLIR op attributes are immutable, each op is replaced in-place with
// a newly built op carrying the derived bits.
//
// This pass is idempotent: running it twice on the same sequence produces the
// same result.

#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// 4-counter semaphore state (faithful to upstream instructions_template.py)
// ---------------------------------------------------------------------------
// Signal naming matches the upstream Python:
//   CMP->LD : compute pushes to load (allows next load group to start)
//   LD->CMP : load pushes to compute (data is ready)
//   CMP->ST : compute pushes to store (output is ready to store)
//   ST->CMP : store pushes to compute (store done, OK to overwrite ACC)
struct Sem {
  int cmpLd = 0, ldCmp = 0, cmpSt = 0, stCmp = 0;
};

// Classify each op into a role for semaphore derivation.
enum class Role {
  LoadInp,    // vtaisa.load buffer_id=INP, y>0 (data load)
  LoadWgt,    // vtaisa.load buffer_id=WGT, y>0
  LoadAcc,    // vtaisa.load buffer_id=ACC, y>0
  LoadUopData,// vtaisa.load buffer_id=UOP, y>0 (gemm/alu uop table data)
  NopInp,     // vtaisa.load buffer_id=INP, y=0 (termination NOP)
  NopUop,     // vtaisa.load buffer_id=UOP, y=0 (termination NOP)
  GemmReset,  // vtaisa.gemm_insn reset=true
  GemmMain,   // vtaisa.gemm_insn reset=false
  AluInsn,    // vtaisa.alu_insn
  StoreOut,   // vtaisa.store
  UopTable,   // vtaisa.uop_table  (no dep bits)
  Finish,     // vtaisa.finish
};

static Role classifyOp(Operation *op) {
  if (auto ld = dyn_cast<vtaisa::LoadOp>(op)) {
    bool isNop = (ld.y_size() == 0);
    switch (ld.buffer_id()) {
    case vtaisa::BufferId::INP: return isNop ? Role::NopInp  : Role::LoadInp;
    case vtaisa::BufferId::WGT: return Role::LoadWgt;
    case vtaisa::BufferId::ACC: return Role::LoadAcc;
    case vtaisa::BufferId::UOP: return isNop ? Role::NopUop  : Role::LoadUopData;
    default: return Role::LoadInp; // fallback
    }
  }
  if (auto g = dyn_cast<vtaisa::GemmInsnOp>(op))
    return g.reset() ? Role::GemmReset : Role::GemmMain;
  if (isa<vtaisa::AluInsnOp>(op))   return Role::AluInsn;
  if (isa<vtaisa::StoreOp>(op))     return Role::StoreOut;
  if (isa<vtaisa::UopTableOp>(op))  return Role::UopTable;
  if (isa<vtaisa::FinishOp>(op))    return Role::Finish;
  return Role::UopTable; // unknown — leave alone
}

// ---------------------------------------------------------------------------
// Rebuild a LoadOp with new dep bits (all other fields unchanged).
// ---------------------------------------------------------------------------
static void replaceLoadDep(OpBuilder &b, vtaisa::LoadOp op,
                            bool pprev, bool pnext, bool qprev, bool qnext) {
  b.setInsertionPoint(op);
  b.create<vtaisa::LoadOp>(op.getLoc(), op.buffer_id(),
                            pprev, pnext, qprev, qnext,
                            op.sram_base(), op.dram_base(),
                            op.y_size(), op.x_size(), op.x_stride());
  op.erase();
}

// ---------------------------------------------------------------------------
// Process one "program" (list of ops between finish boundaries).
// ---------------------------------------------------------------------------
static void deriveProgram(ArrayRef<Operation *> prog) {
  if (prog.empty())
    return;

  // Precompute roles for look-ahead.
  SmallVector<Role> roles;
  roles.reserve(prog.size());
  for (auto *op : prog)
    roles.push_back(classifyOp(op));

  Sem s;
  OpBuilder b(prog[0]->getContext());

  // Helper: is the next non-UopTable op a StoreOut?
  auto nextIsStore = [&](int i) -> bool {
    for (int j = i + 1; j < (int)prog.size(); ++j)
      if (roles[j] != Role::UopTable && roles[j] != Role::LoadUopData)
        return roles[j] == Role::StoreOut;
    return false;
  };

  // Collect consecutive STOREs so we can set first/last dep bits.
  SmallVector<int> storeGroup;

  auto flushStores = [&]() {
    int n = storeGroup.size();
    for (int gi = 0; gi < n; ++gi) {
      int idx = storeGroup[gi];
      auto op = cast<vtaisa::StoreOp>(prog[idx]);
      bool popPrev = (gi == 0) && (s.cmpSt > 0);
      bool pushPrev = (gi == n - 1) && (s.stCmp == 0);
      if (popPrev)  s.cmpSt--;
      if (pushPrev) s.stCmp++;
      b.setInsertionPoint(op);
      b.create<vtaisa::StoreOp>(op.getLoc(), op.buffer_id(),
                                 popPrev, false, pushPrev, false,
                                 op.sram_base(), op.dram_base(),
                                 op.y_size(), op.x_size(), op.x_stride());
      op.erase();
    }
    storeGroup.clear();
  };

  for (int i = 0; i < (int)prog.size(); ++i) {
    Role role = roles[i];
    Operation *rawOp = prog[i];

    // Flush pending STORE group before any non-STORE.
    if (role != Role::StoreOut && !storeGroup.empty())
      flushStores();

    switch (role) {
    case Role::LoadInp: {
      // LD side: pop_next=ack(CMP->LD), push_next=ready(LD->CMP) if no WGT follows.
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool ack   = (s.cmpLd > 0);
      bool ready = (s.ldCmp == 0) && (roles[i+1] != Role::LoadWgt);
      if (ack)   s.cmpLd--;
      if (ready) s.ldCmp++;
      replaceLoadDep(b, op, false, ack, false, ready);
      break;
    }
    case Role::LoadWgt: {
      // LD side: push_next=produce LD->CMP (last data LOAD in step).
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool ready = (s.ldCmp == 0);
      if (ready) s.ldCmp++;
      replaceLoadDep(b, op, false, false, false, ready);
      break;
    }
    case Role::LoadAcc: {
      // Compute side: pop_prev=consume LD->CMP, pop_next=consume ST->CMP.
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool popPrev = (s.ldCmp > 0);
      bool popNext = (s.stCmp > 0);
      if (popPrev) s.ldCmp--;
      if (popNext) s.stCmp--;
      replaceLoadDep(b, op, popPrev, popNext, false, false);
      break;
    }
    case Role::LoadUopData: {
      // Compute side: pop_prev=consume LD->CMP (if no ACC LOAD preceded).
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool popPrev = (s.ldCmp > 0);
      bool popNext = (s.stCmp > 0);
      if (popPrev) s.ldCmp--;
      if (popNext) s.stCmp--;
      replaceLoadDep(b, op, popPrev, popNext, false, false);
      break;
    }
    case Role::NopInp: {
      // Termination NOP-LOAD INP: pop_next=drain CMP->LD, push_next=produce LD->CMP.
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool popNext  = (s.cmpLd > 0);
      bool pushNext = (s.ldCmp == 0);
      if (popNext)  s.cmpLd--;
      if (pushNext) s.ldCmp++;
      replaceLoadDep(b, op, false, popNext, false, pushNext);
      break;
    }
    case Role::NopUop: {
      // Termination NOP-LOAD UOP: pop_prev=drain LD->CMP, pop_next=drain ST->CMP.
      auto op = cast<vtaisa::LoadOp>(rawOp);
      bool popPrev = (s.ldCmp > 0);
      bool popNext = (s.stCmp > 0);
      if (popPrev) s.ldCmp--;
      if (popNext) s.stCmp--;
      replaceLoadDep(b, op, popPrev, popNext, false, false);
      break;
    }
    case Role::GemmReset: {
      // Reset GEMM: push_prev=produce CMP->LD (primes LD<->CMP pipeline).
      auto op = cast<vtaisa::GemmInsnOp>(rawOp);
      bool pushPrev = (s.cmpLd == 0);
      if (pushPrev) s.cmpLd++;
      b.setInsertionPoint(op);
      b.create<vtaisa::GemmInsnOp>(op.getLoc(),
                                    false, false, pushPrev, false,
                                    true, op.uop_bgn(), op.uop_end(),
                                    op.loop_out(), op.loop_in(),
                                    op.dst_factor_out(), op.dst_factor_in(),
                                    op.src_factor_out(), op.src_factor_in());
      op.erase();
      break;
    }
    case Role::GemmMain: {
      // Main GEMM: push_prev=CMP->LD (allow next load), push_next=CMP->ST if STORE follows.
      auto op = cast<vtaisa::GemmInsnOp>(rawOp);
      bool pushPrev = (s.cmpLd == 0);
      bool pushNext = nextIsStore(i) && (s.cmpSt == 0);
      if (pushPrev) s.cmpLd++;
      if (pushNext) s.cmpSt++;
      b.setInsertionPoint(op);
      b.create<vtaisa::GemmInsnOp>(op.getLoc(),
                                    false, false, pushPrev, pushNext,
                                    false, op.uop_bgn(), op.uop_end(),
                                    op.loop_out(), op.loop_in(),
                                    op.dst_factor_out(), op.dst_factor_in(),
                                    op.src_factor_out(), op.src_factor_in());
      op.erase();
      break;
    }
    case Role::AluInsn: {
      // ALU: same semaphore side as GEMM.
      auto op = cast<vtaisa::AluInsnOp>(rawOp);
      bool pushPrev = (s.cmpLd == 0);
      bool pushNext = nextIsStore(i) && (s.cmpSt == 0);
      if (pushPrev) s.cmpLd++;
      if (pushNext) s.cmpSt++;
      b.setInsertionPoint(op);
      b.create<vtaisa::AluInsnOp>(op.getLoc(),
                                   false, false, pushPrev, pushNext,
                                   op.reset(), op.uop_bgn(), op.uop_end(),
                                   op.loop_out(), op.loop_in(),
                                   op.dst_factor_out(), op.dst_factor_in(),
                                   op.src_factor_out(), op.src_factor_in(),
                                   op.alu_opcode(), op.use_imm(), op.imm());
      op.erase();
      break;
    }
    case Role::StoreOut:
      storeGroup.push_back(i);
      break;
    case Role::UopTable:
    case Role::Finish:
      // No dep bits; leave untouched.
      break;
    }
  }

  // Flush any trailing STOREs.
  if (!storeGroup.empty())
    flushStores();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------
struct VTASemaphoreDerivePass
    : public PassWrapper<VTASemaphoreDerivePass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "vta-semaphore-derive"; }
  StringRef getDescription() const final {
    return "Re-derive vtaisa dep bits from the 4-counter semaphore algorithm";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vtaisa::VTAISADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect all vtaisa ops in program order.
    SmallVector<Operation *> all;
    module.walk([&](Operation *op) {
      if (isa<vtaisa::LoadOp, vtaisa::StoreOp, vtaisa::GemmInsnOp,
              vtaisa::AluInsnOp, vtaisa::FinishOp, vtaisa::UopTableOp>(op))
        all.push_back(op);
    });

    if (all.empty())
      return;

    // Split into programs at each vtaisa.finish (inclusive).
    SmallVector<Operation *> prog;
    for (auto *op : all) {
      prog.push_back(op);
      if (isa<vtaisa::FinishOp>(op)) {
        deriveProgram(prog);
        prog.clear();
      }
    }
    if (!prog.empty())
      deriveProgram(prog);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createVTASemaphoreDerivePass() {
  return std::make_unique<VTASemaphoreDerivePass>();
}
