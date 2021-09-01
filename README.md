## cxx inference code

### 0.准备工作

- 配置Clion环境查看 **参考1**；
- 下载libtorch和安装opencv查看 **参考2**
- 拉取submodule
- 阅读参考4，参考5.

### 1.使用test_torch_script.py 转换模型，注意修改对应的配置路径.

### 2.修改配置test/crnn.cpp中的模型路径等.


## 参考
- 1.[Clion Remote debug](https://www.jetbrains.com/help/clion/remote-development.html)
- 2.[PyTorch C++ inference with LibTorch](https://github.com/BIGBALLON/PyTorch-CPP)
- 3.[torch c++ api](https://pytorch.org/docs/stable/cpp_index.html)
- 4.[install c++ distributions of pytorch](https://pytorch.org/cppdocs/installing.html)
- 5.[load a torch script model in c++](https://pytorch.org/tutorials/advanced/cpp_export.html)
- 6.[opencv cmake files install thirdparty](https://github.com/opencv/opencv/blob/master/cmake/OpenCVFindLibsGrfmt.cmake)
- 7.[cmake add third party libraries](https://www.selectiveintellect.net/blog/2016/7/29/using-cmake-to-add-third-party-libraries-to-your-project-1)
- 8.[cnpy read .npy .npz](https://github.com/rogersce/cnpy)
