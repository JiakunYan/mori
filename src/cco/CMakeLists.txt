set(MORI_CCO_SOURCES cco_init.cpp)

add_library(mori_cco SHARED ${MORI_CCO_SOURCES})

target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mori_cco PUBLIC ${CMAKE_SOURCE_DIR})

# SDMA compile switch. PUBLIC so anything linking mori_cco and including cco.hpp
# sees the same BUILD_CCO_SDMA value the library was built with (the header
# defaults it to 0). The JIT device wrapper (compiled out-of-tree by bitcode.py)
# must pass a matching -DBUILD_CCO_SDMA to stay ABI-consistent.
if(BUILD_CCO_SDMA)
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=1)
else()
  target_compile_definitions(mori_cco PUBLIC BUILD_CCO_SDMA=0)
endif()

# Absorb the application layer statically so libmori_cco.so has NO runtime
# dependency on libmori_application.so: --whole-archive pulls in the full
# archive. --exclude-libs,ALL then marks every symbol coming from a static
# archive (mori_application_static, plus the absorbed hsakmt / spdlog / builtins)
# as local, so none are re-exported (no symbol leakage / interposition). cco's
# own cco* symbols live in this target's object file, not an archive, so they
# stay exported. The static archive's INTERFACE brings the external link deps
# (hip/dl/hsa/hsakmt[/mpi]); the ibverbs dlopen shim is embedded, so cco
# links neither libmori_application.so nor libibverbs directly.
target_link_libraries(mori_cco PRIVATE
                      -Wl,--whole-archive mori_application_static -Wl,--no-whole-archive)
target_link_options(mori_cco PRIVATE LINKER:--exclude-libs,ALL)

target_link_libraries(mori_cco PUBLIC mori_logging hip::host)

# ccoProxy (host-proxy RDMA over KIWI invoke). The engine TU is HIP because the
# KIWI headers are. KIWI invoke is header-only on the KIWI queue library, which
# is static and PIC, so it is absorbed like the application layer (its symbols
# stay local via --exclude-libs). The KIWI headers are PUBLIC because
# cco_proxy.hpp includes them; kiwi_queue_headers carries KIWI's include
# directory, which holds the invoke headers too.
# MORI_CCO_PROXY is PUBLIC so kernels including cco_proxy.hpp can tell the
# transport was built in.
if(BUILD_CCO_PROXY)
  target_sources(mori_cco PRIVATE cco_proxy_engine.hip.cpp)
  set_source_files_properties(cco_proxy_engine.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_definitions(mori_cco PUBLIC MORI_CCO_PROXY=1)
  target_link_libraries(mori_cco PUBLIC kiwi_queue_headers PRIVATE KIWI::invoke)
  set_target_properties(mori_cco PROPERTIES HIP_STANDARD 17)
endif()

set_target_properties(
  mori_cco
  PROPERTIES BUILD_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             INSTALL_RPATH "$ORIGIN;${ROCM_PATH}/lib"
             BUILD_WITH_INSTALL_RPATH TRUE)
