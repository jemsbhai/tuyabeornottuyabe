##
# @file CMakeLists.txt
# @brief 
#/

# APP_PATH - 当前CMakeLists.txt所在目录
set(APP_PATH ${CMAKE_CURRENT_LIST_DIR})

# APP_NAME - 从当前目录名获取项目名称
get_filename_component(APP_NAME ${APP_PATH} NAME)

# APP_SRC - 收集src目录下的所有源文件
aux_source_directory(${APP_PATH}/src APP_SRC)

# 收集子目录下的源文件
foreach(dir sd dev_config qmi8658 pcf85063a lc76g wifi_ble mic_spk ui ui/screens ui/fonts ui/images)
    aux_source_directory(${APP_PATH}/src/${dir} SRC)
    list(APPEND APP_SRC ${SRC})
endforeach()

# APP_INC
set(APP_INC 
    ${APP_PATH}/include
    ${APP_PATH}/src/sd
    ${APP_PATH}/src/dev_config
    ${APP_PATH}/src/qmi8658
    ${APP_PATH}/src/pcf85063a
    ${APP_PATH}/src/lc76g
    ${APP_PATH}/src/wifi_ble
    ${APP_PATH}/src/mic_spk
    ${APP_PATH}/src/ui
    ${APP_PATH}/src/ui/screens
    ${APP_PATH}/src/ui/fonts
    ${APP_PATH}/src/ui/images
)

# APP_OPTIONS
set(APP_OPTIONS "-W" "-Wno-maybe-uninitialized")
list(APPEND APP_OPTIONS "-Wall")

########################################
# Target Configure
########################################
add_library(${EXAMPLE_LIB})
message(STATUS "EXAMPLE_LIB:${APP_PATH}")

target_sources(${EXAMPLE_LIB}
    PRIVATE
        ${APP_SRC}
    )

target_include_directories(${EXAMPLE_LIB}
    PRIVATE
        ${APP_INC}
    )

target_compile_options(${EXAMPLE_LIB}
    PRIVATE
        ${APP_OPTIONS}
    )
