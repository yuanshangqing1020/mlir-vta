#pragma once
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace vta {

inline DictionaryAttr findLayerDict(ModuleOp module, StringRef name) {
  if (name.empty())
    return {};
  auto layers = module->getAttrOfType<ArrayAttr>("vta.layers");
  if (!layers)
    return {};
  for (Attribute a : layers) {
    auto d = a.cast<DictionaryAttr>();
    if (d.getAs<StringAttr>("name").getValue() == name)
      return d;
  }
  return {};
}

} // namespace vta
} // namespace mlir
