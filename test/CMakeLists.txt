set(FILE_TEST
    test_main.cc
    test_solver.cc
    test_ezarpack.cc
    )

add_executable(unit_tests
    ${FILE_TEST}
    )

target_link_libraries(unit_tests
    PRIVATE
    fmt-header-only
    sisolver
    )
