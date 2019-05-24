load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

py_binary(
    name = "make_graph",
    srcs = ["make_graph.py"],
        )

tf_library(
    name = "test_graph",
    config = "test_graph_tfmatmulandadd.config.pbtxt",
    cpp_class = "MatMulAndAddComp",
    graph = "test_graph_tfmatmulandadd.pb",
        )

cc_binary(
    name = "run_func",
    srcs = ["use_graph.cc",],
    deps = [
        ":test_graph",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)
