#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace vta {

std::unique_ptr<mlir::Pass> createLowerVTAGemmPass();

std::unique_ptr<mlir::Pass> createLowerVTAAluPass();

std::unique_ptr<mlir::Pass> createConvertLinalgToVTAPass();

std::unique_ptr<mlir::Pass> createVTADramAllocationPass();

std::unique_ptr<mlir::Pass> createVTASemaphoreDerivePass();

void registerVTAPasses();

} // namespace vta
} // namespace mlir
