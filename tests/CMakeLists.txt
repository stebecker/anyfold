INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()



IF(CUDA_FOUND)
CUDA_ADD_EXECUTABLE(test_anyfold test_anyfold.cu)
target_link_libraries(test_anyfold boost_system boost_filesystem boost_unit_test_framework )
set_target_properties(test_anyfold PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include")
ELSE(CUDA_FOUND)
WARNING("Skipping CUDA based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)
