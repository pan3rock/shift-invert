set(FILE_SRC
    shiftinvert_solver.cc
    timer.cc)

add_library(sisolver
    ${FILE_SRC})

target_link_libraries(sisolver
    fmt-header-only)