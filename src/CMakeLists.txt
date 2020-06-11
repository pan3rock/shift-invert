set(FILE_SRC
    shiftinvert_solver.cc
    ezsolver.cc
    timer.cc)

add_library(sisolver
    ${FILE_SRC})

target_link_libraries(sisolver
    ezarpack
    fmt-header-only)