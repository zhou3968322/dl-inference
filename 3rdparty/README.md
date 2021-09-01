# Download zlib to this directory 

```bash
cd dl-inference
git submodule update --init --recursive
```

__增加submodule__

```bash
git submodule add --branch master --force https://github.com/protocolbuffers/protobuf.git 3rdparty/protobuf
git submodule add --branch develop --force https://github.com.cnpmjs.org/nlohmann/json.git 3rdparty/json
```

如果是需要删除某个submodule，需要分别删除submodule路径以及.git/config,以及./git/modules 下面的工程目录.

