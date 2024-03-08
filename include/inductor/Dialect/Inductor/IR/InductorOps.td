#ifndef INDUCTOR_OPS
#define INDUCTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "inductor/Dialect/Inductor/IR/InductorBase.td"
include "inductor/Dialect/Inductor/IR/InductorAttrs.td"
include "inductor/Dialect/Inductor/IR/InductorInterfaces.td"

defvar InductorIndex = I64;

class Inductor_Op<string mnemonic, list<Trait> traits = []>
	: Op<Inductor_Dialect, mnemonic, traits> {
}

/// Python Inductor-IR looks like this:
/// TensorBox -> StorageBox -> Buffer
/// TensorBox -> View -> StorageBox -> Buffer
/// StorageBox -> Loops
/// 1. We skip tensor-box, which contains no information.
/// 2. StorageBoxOp will only contains buffer's layout, and BufferOp contains all things other than layout. And BufferOp's are always nested under StorageBoxOp. This avoids that we have to create `num-of-layout x num-of-buffer-sub-class` kinds of buffer-op.

/// ===== StorageBox =====

/// Constraints: StorageBox must points to a Buffer.
/// What an inductor-ir StorageBox maps to depends on its buffer's layout.
/// FixedLayout/FlexibleLayout: AllocStorageBoxOp
/// AliasedLayout: AliasedStorageBoxOp
/// MutationLayout: MutationStorageBoxOp
class Inductor_StorageOpBase<string mnemonic, list<Trait> traits = []>
	: Inductor_Op<mnemonic, !listconcat(traits, [AttrSizedOperandSegments, StorageOpInterface, NoTerminator, SingleBlock])> {
	dag storage_arguments = (ins);
	/// The following fields are in fact common among inductor-ir Layout.
	dag storage_common_arguments = (ins Builtin_StringAttr:$name, InductorDeviceAttr:$device, TypeAttr:$dtype, Variadic<InductorIndex>:$size, Variadic<InductorIndex>:$stride, InductorIndex:$offset);
	let arguments = !con(storage_common_arguments, storage_arguments);
	/// data_region should only contains one block with one (non-terminator) op, which is the buffer-op.
	dag regions = (region SizedRegion<1>:$data_region);
	/// result represents the allocated buffer or aliased buffer.
	dag results = (outs AnyMemRef:$result);

  	let skipDefaultBuilders = 1;
  	let builders = [
		OpBuilder<(ins "::mlir::Type":$result, "::mlir::StringAttr":$name, "::mlir::inductor::DeviceAttr":$device, "::mlir::TypeAttr":$dtype, "::mlir::ValueRange":$size, "::mlir::ValueRange":$stride, "::mlir::Value":$offset, "/*optional*/::mlir::UnitAttr":$is_fixed)>,
  	];
}

def Inductor_PlainStorageOp: Inductor_StorageOpBase<"plain_storage", []> {
	/// The presence of is_fixed means the storage's layout is fixed, otherwise flexible.
	let storage_arguments = (ins UnitAttr:$is_fixed);
}

def Inductor_AliasedStorageOp: Inductor_StorageOpBase<"aliased_storage", []> {
	/// view is the view-op that this storage aliased.
	dag storage_arguments = (ins AnyMemRef:$view);
}

def Inductor_MutationStorageOp: Inductor_StorageOpBase<"mutation_storage", []> {
	/// target is the tensor that this op mutates.
	dag storage_arguments = (ins AnyMemRef:$target);
}

/// ===== Buffer =====
class Inductor_BufferOpBase<string mnemonic, list<Trait> traits>: Inductor_Op<mnemonic, !listconcat(traits, [BufferOpInterface, NoTerminator, SingleBlock])> {
	dag buffer_arguments = (ins);
	dag buffer_regions = (region);

	dag buffer_common_arguments = (ins OptionalAttr<Builtin_StringAttr>:$name);
	let arguments = !con(buffer_common_arguments, buffer_arguments);
	let regions = buffer_regions;
}

def Inductor_InputBufferOp : Inductor_BufferOpBase<"input_buffer", []>;

def Inductor_ConstantBufferOp : Inductor_BufferOpBase<"constant_buffer", []> {
	let buffer_arguments = (ins OptionalAttr<InductorDeviceAttr>:$override_device);
}
// TODO: NoneAsConstantBuffer, ShapeAsConstantBuffer
def Inductor_ComputedBufferOp : Inductor_BufferOpBase<"computed_buffer", []> {
	let buffer_regions = (region SizedRegion<1>:$dataRegion);
	let extraClassDeclaration = [{
		::mlir::inductor::LoopsOpInterface getLoop();
	}];
  	let skipDefaultBuilders = 1;
  	let builders = [
		OpBuilder<(ins "/*optional*/::mlir::StringAttr":$name)>,
  	];
}
// FIXME: support template buffer
// FIXME: TritonTemplateBuffer/CUDATemplateBuffer
def Inductor_TemplateBufferOp : Inductor_BufferOpBase<"template_buffer", []>;
// FIXME: support InputsKernel
def Inductor_InputsKernelOp : Inductor_BufferOpBase<"inputs_kernel", []>;


//Buffer
//	InputBuffer
//		ConstantBuffer
//	ComputedBuffer
//	TemplateBuffer
//		TritonTemplateBuffer
//		CudaTemplateBuffer
//	InputsKernel
//		NopKernel
//			ConcatKernel
//		ExternKernel
//			ExternKernelOut
//				RandomSeeds
//				DeviceCopy
//			ExternKernelAloc
//				FallbackKernel
//					ComplexView
//					_CollectiveKernel
//						_WaitKernel
//				ConvolutionUnary
//				ConvolutionBinary
//				ConvolutionBinaryInplace
//				MKLPackedLinear
//				LinearUnary
//				LinearBinary
//				ConvolutionTransposeUnary
//				MkldnnRnnLayer
//				QConvPointWisePT2E
//				QConvPointWiseBinaryPT2E
//				QLinearPointwisePT2E
//				Wait
//			CollectiveKernel
//				InplaceBernoulliFallback
//				OutOfPlaceCollectiveKernel
//					AllGatherIntoTensor
//					ReduceScatterTensor
//					AllGatherIntoTensorCoalesced
//					ReduceScatterTensorCoalesced
//					AllToAllSingle
//				InPlaceCollectiveKernel
//					Broadcast
//					AllReduceCoalesced
//					AllReduce
//			InPlaceHint
//			OutputBuffer
//			UserDefinedTritonKernel
//			MutationOutput
//			InplaceBernoulliFallback
//			InplaceCopyFallback
//			MutatingFirstArgExternKernel
//				ResizeStorageBytes
//			ScatterFallback
//			IndexPutFallback
//			DynamicScalar
//			AssertScalar
//			MultiOutput
//				MultiOutputNoSizeAssert
//			Conditional

/// Base class for ops that only appear in inner_fn, a.k.a. what needed by ops_handler.
class Inductor_OpsOpBase<string mnemonic, list<Trait> traits = []>
	: Inductor_Op<mnemonic, traits> {
}

def Inductor_CalcPlainOffsetOp : Inductor_OpsOpBase<"ops_calc_plain_offset", [SameVariadicOperandSize]> {
	let arguments = (ins Variadic<InductorIndex>:$coord, Variadic<InductorIndex>:$size, Variadic<InductorIndex>:$stride, InductorIndex:$offset);
	let results = (outs InductorIndex:$result);
}

def Inductor_LoadOp : Inductor_OpsOpBase<"ops_load", []> {
	let arguments = (ins Builtin_StringAttr:$buffer, InductorIndex:$offset);
	let results = (outs AnyType:$result);
}

def Inductor_YieldOp : Inductor_OpsOpBase<"ops_yield", [Terminator]> {
	let arguments = (ins AnyType:$value);
}

class Inductor_LoopsOpBase<string mnemonic, list<Trait> traits>
	: Inductor_Op<mnemonic, !listconcat(traits, [LoopsOpInterface])> {
	/// Following fileds should be filled by sub classes.
	dag loops_arguments = (ins);
	dag loops_regions = (region);

	/// Following fileds are from Loops base class.
	dag loops_common_arguments = (ins InductorDeviceAttr:$device, Variadic<InductorIndex>:$ranges);
	dag loops_common_regions = (region AnyRegion:$inner_fn);

	/// Combine loops base-class fileds and sub-classe fields.
	let arguments = !con(loops_common_arguments, loops_arguments);
	let regions = !con(loops_common_regions, loops_regions);
}

class Inductor_PointwiseOpBase<string mnemonic, list<Trait> traits>
	: Inductor_LoopsOpBase<mnemonic, traits> {
	/// Following fileds should be filled by sub classes.
	dag pointwise_arguments = (ins);
	dag pointwise_regions = (region);

	/// Inheritance boilerpolates.
	/// This class adds nothing.
	let loops_arguments = pointwise_arguments;
	let loops_regions = pointwise_regions;
}

def Inductor_PointwiseOp : Inductor_PointwiseOpBase<"pointwise", []>;

def Inductor_ScatterOp : Inductor_PointwiseOpBase<"scatter", []> {
	let pointwise_arguments = (ins Builtin_StringAttr:$scatter_mode);
	let pointwise_regions = (region AnyRegion:$output_index);
}

class Inductor_ReductionOpBase<string mnemonic, list<Trait> traits>
	: Inductor_LoopsOpBase<mnemonic, !listconcat(traits, [AttrSizedOperandSegments])> {
	/// Fields to be filled by child class.
	dag reduction_arguments = (ins);

	/// Common reduction fields.
	dag reduction_common_arguments = (ins Variadic<InductorIndex>:$reduction_ranges, Builtin_StringAttr:$reduction_type, InductorReductionHintAttr:$reduction_hint);

	/// Inheritance boilerpolates.
	let loops_arguments = !con(reduction_common_arguments, reduction_arguments);
}

def Inductor_ReductionOp : Inductor_ReductionOpBase<"reduction", []> {
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::inductor::DeviceAttr":$device, "::mlir::ValueRange":$ranges, "::mlir::ValueRange":$reduction_ranges, "::mlir::StringAttr":$reduction_type, "::mlir::inductor::ReductionHintAttr":$reduction_hint)>,
  ];
}

def Inductor_WelfordReductionOp : Inductor_ReductionOpBase<"welford_reduction", []> {
	let reduction_arguments = (ins InductorIndex:$output_index);
}

class Inductor_ScanOpBase<string mnemonic, list<Trait> traits>
	: Inductor_LoopsOpBase<mnemonic, !listconcat(traits, [AttrSizedOperandSegments])> {
	/// Common scan fields.
	dag scan_common_arguments = (ins Variadic<InductorIndex>:$scan_ranges, Variadic<InductorIndex>:$size, InductorReductionHintAttr:$reduction_hint, I64:$init);
	dag scan_common_regions = (region AnyRegion:$combine_fn, AnyRegion:$reindex);

	/// Inheritance boilerpolates.
	let loops_arguments = scan_common_arguments;
	let loops_regions = scan_common_regions;
}

def Inductor_ScanOp : Inductor_ScanOpBase<"scan", []>;
def Inductor_SplitScanOp : Inductor_ScanOpBase<"split_scan", []>;

class Inductor_BaseViewOpBase<string mnemonic, list<Trait> traits>
	: Inductor_Op<mnemonic, traits> {
	dag base_view_arguments = (ins);
	dag base_view_regions = (region);

	dag base_view_common_arguments = (ins AnyMemRef:$data);

	let arguments = !con(base_view_common_arguments, base_view_arguments);
	let regions = base_view_regions;
	let results = (outs AnyMemRef:$result);
}

def Inductor_ExpandViewOp : Inductor_BaseViewOpBase<"expand_view", []> {
	let base_view_arguments = (ins Variadic<InductorIndex>:$size);
}

def Inductor_PermuteViewOp : Inductor_BaseViewOpBase<"permute_view", []> {
	let base_view_arguments = (ins Variadic<InductorIndex>:$dims);
}

def Inductor_SqueezeViewOp : Inductor_BaseViewOpBase<"squeeze_view", []>;

class Inductor_ViewOpBase<string mnemonic, list<Trait> traits> : Inductor_BaseViewOpBase<mnemonic, traits> {
	dag generic_view_arguments = (ins);

	dag generic_view_common_arguments = (ins Variadic<InductorIndex>:$size);
	dag generic_view_common_regions = (region AnyRegion:$reindex);

	let base_view_arguments = !con(generic_view_common_arguments, generic_view_arguments);
	let base_view_regions = generic_view_common_regions;
}

def Inductor_ViewOp : Inductor_ViewOpBase<"view", []>;
def Inductor_SliceViewOp : Inductor_ViewOpBase<"slice_view", []>;

def Inductor_ReinterpretViewOp : Inductor_BaseViewOpBase<"reinterpret_view", []>;

/// ===== Constant =====
class Inductor_ConstantOpBase<string mnemonic, list<Trait> traits> : Inductor_Op<mnemonic, traits> {
}

def Inductor_IndexingConstantOp : Inductor_ConstantOpBase<"indexing_constant", []>;


//
//
//MultiOutputLayout
//
//ExternKernelNode
//
//	NoneAsConstantBuffer
//	ShapeAsConstantBuffer
//
//
//Subgraph
//
//LoopBody
//LoopBodyBlock

#endif // INDUCTOR_OPS
