package = "Async-EASGD"
version = "scm-1"

source = {
   url = "git://github.com/shanlior/Async-EASGD.git",
}

description = {
   summary = "An asynchronous inhomogeneous implementation of Elastic-Averaging EASGD",
   homepage = "-",
   license = "MIT",
}

dependencies = {
   "torch >= 7.0",
   "regress",
   "dataset",
   "ipc",
}

build = {
   type = "builtin",
   modules = {
      ['Async-EASGD.AsyncEA'] = 'lua/AsyncEA.lua',
      ['colorPrint'] = 'lua/colorPrint.lua'
   },
}
