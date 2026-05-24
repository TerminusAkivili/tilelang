/*!
 * \file tl/op/packed_lowbit.h
 * \brief Shared predicates for semantic low-bit dtypes with carrier storage.
 *
 * This keeps the public dtype semantic while allowing SM120 FP4 lowering to
 * use the byte-container shared-memory shape consumed by ordinary ldmatrix.
 */

#ifndef TVM_TL_OP_PACKED_LOWBIT_H_
#define TVM_TL_OP_PACKED_LOWBIT_H_

#include "op/copy.h"
#include "op/utils.h"
#include "target/utils.h"
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace packed_lowbit {

// Function attribute listing semantic FP4 operands whose physical storage is
// treated as byte-carrier inside the SM120 MMA lowering path.
constexpr const char *kByteCarrierBuffersAttr =
    "tl.packed_lowbit_byte_carrier_buffers";
constexpr const char *kByteCarrierCopyAttr =
    "tl.packed_lowbit_byte_carrier_copy";

inline bool IsFp4E2M1(DataType dtype) {
  return dtype.is_float4_e2m1fn() && dtype.is_scalar();
}

inline bool IsFp4DTypeString(const std::string &dtype) {
  return dtype == "float4_e2m1fn" || dtype == "e2m1";
}

inline bool HasFp4MmaOperand(const tirx::Stmt &stmt) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    const auto *call = node.as<tirx::CallNode>();
    if (call == nullptr || !call->op.same_as(tirx::builtin::ptx_mma())) {
      return;
    }
    ICHECK_GE(call->args.size(), 5U);
    std::string a_dtype = Downcast<tirx::StringImm>(call->args[3])->value;
    std::string b_dtype = Downcast<tirx::StringImm>(call->args[4])->value;
    found = IsFp4DTypeString(a_dtype) || IsFp4DTypeString(b_dtype);
  });
  return found;
}

inline bool IsGlobalLikeBuffer(const Buffer &buffer,
                               bool allow_empty_scope = false) {
  return IsGlobalBuffer(buffer) ||
         (allow_empty_scope && buffer.defined() && buffer.scope().empty());
}

inline bool IsSM120Fp4GlobalToSharedCopy(const CopyNode &op, Target target,
                                         bool allow_empty_global = false) {
  return TargetIsSM120(target) && IsFp4E2M1(op.src->dtype) &&
         op.src->dtype == op.dst->dtype &&
         IsGlobalLikeBuffer(op.src, allow_empty_global) &&
         IsSharedBuffer(op.dst);
}

inline bool IsSM120Fp4GlobalSharedCopy(const CopyNode &op, Target target,
                                       bool allow_empty_global = false) {
  if (IsSM120Fp4GlobalToSharedCopy(op, target, allow_empty_global)) {
    return true;
  }
  return TargetIsSM120(target) && IsFp4E2M1(op.dst->dtype) &&
         op.src->dtype == op.dst->dtype && IsSharedBuffer(op.src) &&
         IsGlobalLikeBuffer(op.dst, allow_empty_global);
}

inline bool HasByteCarrierCopyAttr(const CopyNode &op) {
  if (auto val = op.annotations.Get(kByteCarrierCopyAttr)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

inline bool RequiresSM120Fp4PackedLowbitCopy(
    const CopyNode &op, Target target, bool allow_empty_global = false) {
  return IsSM120Fp4GlobalSharedCopy(op, target, allow_empty_global) &&
         !HasByteCarrierCopyAttr(op);
}

inline std::unordered_set<std::string>
GetByteCarrierBuffers(const tirx::PrimFunc &func) {
  std::unordered_set<std::string> buffers;
  if (auto opt = func->GetAttr<Array<String>>(kByteCarrierBuffersAttr)) {
    for (const auto &name : opt.value()) {
      buffers.insert(static_cast<std::string>(name));
    }
  }
  return buffers;
}

inline std::unordered_set<std::string>
CollectSM120Fp4GlobalSharedCopyBuffers(const tirx::Stmt &stmt, Target target,
                                       bool allow_empty_global = false) {
  std::unordered_set<std::string> buffers;
  if (!TargetIsSM120(target)) {
    return buffers;
  }
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    const auto *call = node.as<tirx::CallNode>();
    if (call == nullptr) {
      return;
    }
    TileOperator tile_op = ParseOperator(tvm::ffi::GetRef<tirx::Call>(call));
    const auto *copy = tile_op.as<CopyNode>();
    if (copy == nullptr ||
        !IsSM120Fp4GlobalSharedCopy(*copy, target, allow_empty_global)) {
      return;
    }
    auto add_buffer = [&](const Buffer &buffer) {
      if (IsFp4E2M1(buffer->dtype) &&
          (IsGlobalLikeBuffer(buffer, allow_empty_global) ||
           IsSharedBuffer(buffer))) {
        buffers.insert(buffer->data->name_hint);
      }
    };
    add_buffer(copy->src);
    add_buffer(copy->dst);
  });
  return buffers;
}

} // namespace packed_lowbit
} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_PACKED_LOWBIT_H_
