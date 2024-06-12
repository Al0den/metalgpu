cmake_minimum_required(VERSION 3.23)
project(
	metalgpucpp-arm
	DESCRIPTION "C++ backend for metalgpu python lib"
)

# enable to get commands helping lsp to find libs
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

include_directories("src")

find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)
find_library(COCOA_LIBRARY Cocoa REQUIRED)

file(GLOB SRC_FILES CONFIGURE_DEPENDS
	"src/*.cpp" 
	"src/api/*.cpp" 
	"src/instance/*.cpp"
)

add_library(metalgpucpp-arm SHARED ${SRC_FILES})

target_include_directories(metalgpucpp-arm
  SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/metal-cpp
)

target_link_libraries(metalgpucpp-arm
  ${APPLE_FWK_FOUNDATION}
  ${APPLE_FWK_METAL}
  ${COCOA_LIBRARY}
)

set(CMAKE_INSTALL_PREFIX  "../metalgpu/src/metalgpu/")
install(TARGETS metalgpucpp-arm)
