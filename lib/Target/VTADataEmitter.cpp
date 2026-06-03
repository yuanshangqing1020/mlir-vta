#include "mlir-vta/Target/VTADataEmitter.h"
#include "mlir-vta/VTAGemmLayout.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

constexpr unsigned kDim = 16;                       // block side
constexpr size_t kBlockElems = kDim * kDim;         // 256
constexpr size_t kBlockBytes = kBlockElems * sizeof(int32_t); // 1024

// Read exactly rows*cols int32 from path into a row-major buffer.
static LogicalResult readMatrix(llvm::StringRef path, unsigned rows,
                                unsigned cols, std::vector<int32_t> &out) {
  std::ifstream is(path.str(), std::ios::binary);
  if (!is) {
    llvm::errs() << "vta-translate: cannot open " << path << "\n";
    return failure();
  }
  size_t elems = static_cast<size_t>(rows) * cols;
  size_t bytes = elems * sizeof(int32_t);
  out.resize(elems);
  is.read(reinterpret_cast<char *>(out.data()), bytes);
  if (static_cast<size_t>(is.gcount()) != bytes) {
    llvm::errs() << "vta-translate: " << path << " is not " << bytes
                 << " bytes (" << rows << "x" << cols << " int32)\n";
    return failure();
  }
  return success();
}

static LogicalResult writeBinary(llvm::StringRef path,
                                 const std::vector<int32_t> &data) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os) {
    llvm::errs() << "vta-translate: cannot open " << path << "\n";
    return failure();
  }
  os.write(reinterpret_cast<const char *>(data.data()),
           data.size() * sizeof(int32_t));
  if (!os) {
    llvm::errs() << "vta-translate: error writing " << path << "\n";
    return failure();
  }
  return success();
}

static LogicalResult writeText(llvm::StringRef path, const std::string &text) {
  std::ofstream os(path.str(), std::ios::binary);
  if (!os) {
    llvm::errs() << "vta-translate: cannot open " << path << "\n";
    return failure();
  }
  os.write(text.data(), text.size());
  if (!os) {
    llvm::errs() << "vta-translate: error writing " << path << "\n";
    return failure();
  }
  return success();
}

} // namespace

LogicalResult mlir::vta::emitData(llvm::StringRef inputPath,
                                  llvm::StringRef weightPath,
                                  llvm::StringRef outDir, unsigned mRows,
                                  unsigned kDimElems, unsigned nCols) {
  if (mRows % kDim || kDimElems % kDim || nCols % kDim || mRows == 0 ||
      kDimElems == 0 || nCols == 0) {
    llvm::errs() << "vta-translate: data dims must be positive multiples of 16 "
                    "(got "
                 << mRows << "x" << kDimElems << "x" << nCols << ")\n";
    return failure();
  }
  const unsigned Mb = mRows / kDim, Kb = kDimElems / kDim, Nb = nCols / kDim;

  std::vector<int32_t> input, weight;
  if (failed(readMatrix(inputPath, mRows, kDimElems, input)) ||
      failed(readMatrix(weightPath, kDimElems, nCols, weight)))
    return failure();

  // input.bin: block-major (row-major over blocks), each 16x16 block row-major,
  // no transpose. Block (bi, bk) covers input[16*bi + r][16*bk + c].
  std::vector<int32_t> inBlocked;
  inBlocked.reserve(static_cast<size_t>(mRows) * kDimElems);
  for (unsigned bi = 0; bi < Mb; ++bi)
    for (unsigned bk = 0; bk < Kb; ++bk)
      for (unsigned r = 0; r < kDim; ++r)
        for (unsigned c = 0; c < kDim; ++c)
          inBlocked.push_back(input[(16 * bi + r) * kDimElems + (16 * bk + c)]);
  if (failed(writeBinary((outDir + "/input.bin").str(), inBlocked)))
    return failure();

  // weight.bin: block-major, each 16x16 block transposed. Written block (bk, bj)
  // element (r, c) = weight[16*bk + c][16*bj + r].
  std::vector<int32_t> wBlocked;
  wBlocked.reserve(static_cast<size_t>(kDimElems) * nCols);
  for (unsigned bk = 0; bk < Kb; ++bk)
    for (unsigned bj = 0; bj < Nb; ++bj)
      for (unsigned r = 0; r < kDim; ++r)
        for (unsigned c = 0; c < kDim; ++c)
          wBlocked.push_back(weight[(16 * bk + c) * nCols + (16 * bj + r)]);
  if (failed(writeBinary((outDir + "/weight.bin").str(), wBlocked)))
    return failure();

  // accumulator.bin / out_init.bin: (Mb*Nb) blocks of 256 int32 zeros.
  std::vector<int32_t> zeros(static_cast<size_t>(Mb) * Nb * kBlockElems, 0);
  if (failed(writeBinary((outDir + "/accumulator.bin").str(), zeros)) ||
      failed(writeBinary((outDir + "/out_init.bin").str(), zeros)))
    return failure();

  // The "Is it square?" column is a fixed per-role flag in the upstream
  // compiler (data_definition.py): A/Y/BS = True, X = doExpandBias (False for
  // pure GEMM), C = doStoreFullMatrix (True for pure GEMM). It is NOT derived
  // from the actual dimensions, so rectangular matrices still report True.
  // Python's csv writer terminates rows with CRLF, so reproduce \r\n exactly.
  std::string metadata =
      "Matrix (or Block Size),Nb rows,Nb columns,Is it square?\r\n"
      "BS,16,16,True\r\n";
  metadata += "A," + std::to_string(mRows) + "," + std::to_string(kDimElems) +
              ",True\r\n";
  metadata += "X," + std::to_string(mRows) + "," + std::to_string(nCols) +
              ",False\r\n";
  metadata += "Y,0,0,True\r\n";
  metadata += "C," + std::to_string(mRows) + "," + std::to_string(nCols) +
              ",True\r\n";

  // memory_addresses.csv: page-aligned bases that shift with matrix size
  // (computed identically to the lowering and the upstream dram_allocation).
  const vta::GemmLayout L = vta::computeGemmLayout(Mb, Kb, Nb);
  auto hx = [](int64_t v) {
    char buf[20];
    std::snprintf(buf, sizeof(buf), "0x%llx", (unsigned long long)v);
    return std::string(buf);
  };
  std::string memoryAddresses =
      "Buffer type,Physical address (hex),Logical address (hex)\r\n";
  memoryAddresses += "INP," + hx(L.inpPhys) + "," + hx(L.inpLogical) + "\r\n";
  memoryAddresses += "WGT," + hx(L.wgtPhys) + "," + hx(L.wgtLogical) + "\r\n";
  memoryAddresses += "ACC," + hx(L.accPhys) + "," + hx(L.accLogical) + "\r\n";
  memoryAddresses += "OUT," + hx(L.outPhys) + "," + hx(L.outLogical) + "\r\n";
  memoryAddresses += "UOP," + hx(L.uopPhys) + "," + hx(L.uopLogical) + "\r\n";
  memoryAddresses += "INSN," + hx(L.insnPhys) + "," + hx(L.insnLogical) + "\r\n";

  std::string layersName =
      "Line identifier,Nb of VTA IR,Provide execution log\r\n"
      "nb_vta_ir,1,True\r\n"
      "Line identifier,VTA IR name,Last physical DRAM address allocated by the "
      "layer\r\n";
  layersName += "0,," + hx(L.lastPhys) + "\r\n";

  if (failed(writeText((outDir + "/metadata.csv").str(), metadata)) ||
      failed(writeText((outDir + "/memory_addresses.csv").str(),
                       memoryAddresses)) ||
      failed(writeText((outDir + "/layers_name.csv").str(), layersName)))
    return failure();

  return success();
}
