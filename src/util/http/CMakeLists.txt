add_subdirectory(HttpParser)
add_library(mediaTypes MediaTypes.h MediaTypes.cpp)
target_compile_options(mediaTypes PUBLIC -Wno-attributes)
qlever_target_link_libraries(mediaTypes util httpParser)

if (NOT REDUCED_FEATURE_SET_FOR_CPP17)
add_library(http HttpServer.h HttpServer.cpp HttpClient.h HttpClient.cpp HttpUtils.h HttpUtils.cpp UrlParser.h UrlParser.cpp "HttpParser/AcceptHeaderQleverVisitor.h"
        websocket/WebSocketSession.cpp websocket/MessageSender.cpp websocket/QueryToSocketDistributor.cpp
        websocket/QueryHub.cpp websocket/UpdateFetcher.cpp)

qlever_target_link_libraries(http parser util mediaTypes httpParser OpenSSL::SSL OpenSSL::Crypto)
# NOTE: the GCC-13-on-MinGW SEH coroutine codegen workaround
# (-fno-reorder-blocks-and-partition) that this library used to set on itself is
# now applied build-wide in the top-level CMakeLists.txt, because the bug can hit
# any coroutine translation unit (Server, IndexRebuilder, ...), not just this one.
endif()
