#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace vta {

std::unique_ptr<mlir::Pass> createLowerVTAGemmPass();

std::unique_ptr<mlir::Pass> createConvertLinalgToVTAPass();

void registerVTAPasses();

} // namespace vta
} // namespace mlir
