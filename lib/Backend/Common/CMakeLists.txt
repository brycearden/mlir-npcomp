add_npcomp_library(NPCOMPCommonBackend
  VerifyBackendContract.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SRC_DIR}/include/npcomp/Backend/Common

  DEPENDS
  NPCOMPCommonBackendPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
  MLIRMemRef
  MLIRStandard
  MLIRMath
  )

mlir_check_all_link_libraries(NPCOMPCommonBackend)
