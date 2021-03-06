set(dialect BasicpyOps)
set(dialect_namespace basicpy)
set(LLVM_TARGET_DEFINITIONS ${dialect}.td)

mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIR${dialect}IncGen)
add_dependencies(mlir-headers MLIR${dialect}IncGen)

add_mlir_doc(BasicpyDialect BasicpyDialect Basicpy/ -gen-dialect-doc)
add_mlir_doc(BasicpyOps BasicpyOps Basicpy/ -gen-op-doc)
