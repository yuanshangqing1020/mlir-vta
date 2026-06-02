#include "mlir-vta/Target/VTADataEmitter.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

constexpr unsigned kDim = 16;
constexpr size_t kElems = kDim * kDim;       // 256
constexpr size_t kBytes = kElems * sizeof(int32_t); // 1024

// Read exactly kBytes from path into a 16x16 int32 buffer.
static LogicalResult readMatrix(llvm::StringRef path,
                                std::vector<int32_t> &out) {
  std::ifstream is(path.str(), std::ios::binary);
  if (!is) {
    llvm::errs() << "vta-translate: cannot open " << path << "\n";
    return failure();
  }
  out.resize(kElems);
  is.read(reinterpret_cast<char *>(out.data()), kBytes);
  if (static_cast<size_t>(is.gcount()) != kBytes) {
    llvm::errs() << "vta-translate: " << path << " is not " << kBytes
                 << " bytes\n";
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

static LogicalResult writeText(llvm::StringRef path, llvm::StringRef text) {
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
                                  llvm::StringRef outDir) {
  std::vector<int32_t> input, weight;
  if (failed(readMatrix(inputPath, input)) ||
      failed(readMatrix(weightPath, weight)))
    return failure();

  // input.bin: raw copy for a single block.
  if (failed(writeBinary((outDir + "/input.bin").str(), input)))
    return failure();

  // weight.bin: transpose W[r][c] -> out[c][r].
  std::vector<int32_t> weightT(kElems);
  for (unsigned r = 0; r < kDim; ++r)
    for (unsigned c = 0; c < kDim; ++c)
      weightT[c * kDim + r] = weight[r * kDim + c];
  if (failed(writeBinary((outDir + "/weight.bin").str(), weightT)))
    return failure();

  // accumulator.bin / out_init.bin: 256 int32 zeros.
  std::vector<int32_t> zeros(kElems, 0);
  if (failed(writeBinary((outDir + "/accumulator.bin").str(), zeros)) ||
      failed(writeBinary((outDir + "/out_init.bin").str(), zeros)))
    return failure();

  // Python's csv writer terminates rows with CRLF, so reproduce \r\n exactly.
  static const char *kMetadata =
      "Matrix (or Block Size),Nb rows,Nb columns,Is it square?\r\n"
      "BS,16,16,True\r\n"
      "A,16,16,True\r\n"
      "X,16,16,False\r\n"
      "Y,0,0,True\r\n"
      "C,16,16,True\r\n";
  static const char *kMemoryAddresses =
      "Buffer type,Physical address (hex),Logical address (hex)\r\n"
      "INP,0x1000,0x40\r\n"
      "WGT,0x2000,0x8\r\n"
      "ACC,0x3000,0xc0\r\n"
      "OUT,0x4000,0x100\r\n"
      "UOP,0x5000,0x1400\r\n"
      "INSN,0x6000,0x600\r\n";
  static const char *kLayersName =
      "Line identifier,Nb of VTA IR,Provide execution log\r\n"
      "nb_vta_ir,1,True\r\n"
      "Line identifier,VTA IR name,Last physical DRAM address allocated by the "
      "layer\r\n"
      "0,,0x60af\r\n";

  if (failed(writeText((outDir + "/metadata.csv").str(), kMetadata)) ||
      failed(writeText((outDir + "/memory_addresses.csv").str(),
                       kMemoryAddresses)) ||
      failed(writeText((outDir + "/layers_name.csv").str(), kLayersName)))
    return failure();

  return success();
}
