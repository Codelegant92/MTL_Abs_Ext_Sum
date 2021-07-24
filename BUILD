package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//CNN_Abs_sum_HierAttn_dec_attn_tag_lambda100_beta05/...",
    ],
)

py_library(
    name = "seq2seq_attention_model",
    srcs = ["seq2seq_attention_model.py"],
    deps = [
        ":seq2seq_lib_new",
    ],
)

py_library(
    name = "seq2seq_lib_new",
    srcs = ["seq2seq_lib_new.py"],
)

py_binary(
    name = "seq2seq_attention",
    srcs = ["seq2seq_attention.py"],
    deps = [
        ":batch_reader",
        ":data",
        ":seq2seq_attention_decode",
        ":seq2seq_attention_model",
    ],
)

py_library(
    name = "batch_reader",
    srcs = ["batch_reader.py"],
    deps = [
        ":data",
        ":seq2seq_attention_model",
    ],
)

py_library(
    name = "beam_search",
    srcs = ["beam_search.py"],
)

py_library(
    name = "seq2seq_attention_decode",
    srcs = ["seq2seq_attention_decode.py"],
    deps = [
        ":beam_search",
        ":data",
    ],
)

py_library(
    name = "data",
    srcs = ["data.py"],
)
