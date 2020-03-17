
filegroup(
  name = "libtensorrt_module.so_release", 
  srcs = ["libtensorrt_module.so"],
  visibility = ["//visibility:public"],
)
 
filegroup(
  name = "libtorch_module.so_release", 
  srcs = ["libtorch_module.so"],
  visibility = ["//visibility:public"],
)
 
filegroup(
  name = "pyml.so_release", 
  srcs = ["pyml.so"],
  visibility = ["//visibility:public"],
)
 
filegroup(
  name = "libml_module.so_release", 
  srcs = ["libml_module.so"],
  visibility = ["//visibility:public"],
)
 
filegroup(
  name = "libtensorflow_module.so_release", 
  srcs = ["libtensorflow_module.so"],
  visibility = ["//visibility:public"],
)
 
py_library( 
name = "pyml", 
srcs = ["__init__.py"], 
visibility = ["//visibility:public"], 
data = [":pyml.so_release"], 
imports = [":pyml.so_release"], 
)