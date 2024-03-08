#ifndef INDUCTOR_IR_INDUCTOR_INTERFACES
#define INDUCTOR_IR_INDUCTOR_INTERFACES

include "mlir/IR/OpBase.td"

def BufferOpInterface: OpInterface<"BufferOpInterface"> {
  let description = [{
  	Interface for Ops which represents inductor-ir Buffer.
  }];
  let cppNamespace = "::mlir::inductor";
}

def LoopsOpInterface: OpInterface<"LoopsOpInterface"> {
  let description = [{
  	Interface for Ops which represents inductor-ir Loops.
  }];
  let cppNamespace = "::mlir::inductor";
}

def StorageOpInterface : OpInterface<"StorageOpInterface"> {
  let description = [{
  	Interface for Ops which represents inductor-ir Storage.
  }];
  let cppNamespace = "::mlir::inductor";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the only data op contained in this storage-box",
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"getDataOp",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
      	return ::mlir::inductor::impl::getDataOpFromDataRegion($_op.getDataRegion());
      }]
      >,
    InterfaceMethod<
      /*desc=*/"Returns dtype as MLIR type",
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getDtype",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
      >,
    InterfaceMethod<
      /*desc=*/"Returns name of the buffer",
      /*retTy=*/"::llvm::StringRef",
      /*methodName=*/"getName",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{}]
      >
  ];
}

#endif
