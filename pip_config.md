# Pip 镜像源配置指南

## 已配置的镜像源
当前已配置清华大学镜像源：`https://pypi.tuna.tsinghua.edu.cn/simple`

## 常用国内镜像源

### 1. 清华大学镜像源（推荐）
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2. 阿里云镜像源
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com
```

### 3. 中国科技大学镜像源
```bash
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/
pip config set global.trusted-host pypi.mirrors.ustc.edu.cn
```

### 4. 豆瓣镜像源
```bash
pip config set global.index-url https://pypi.douban.com/simple/
pip config set global.trusted-host pypi.douban.com
```

## 临时使用镜像源
如果不想永久配置，可以在安装时临时指定：

```bash
pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 查看当前配置
```bash
pip config list
```

## 重置为默认源
```bash
pip config unset global.index-url
pip config unset global.trusted-host
```

## 其他加速技巧

### 1. 使用缓存
```bash
pip install --cache-dir /path/to/cache package_name
```

### 2. 并行下载
```bash
pip install -U pip  # 升级pip到最新版本支持并行下载
```

### 3. 预编译包
```bash
pip install --only-binary=all package_name  # 只下载预编译包
```

### 4. 使用conda镜像（如果使用conda）
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
``` 