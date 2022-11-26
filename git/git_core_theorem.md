git有4个工作区域，

工作目录区域,本地空间（working directory)：
你平时放东西的地方

缓存区（stage/index)：
实际上只是一个文件，临时存放你的改动

资源库,本地仓库（local repo）:
安全存放数据的地方，HEAD文件指向最新放入仓库的版本，.git属于隐藏文件

远程的仓库(remote git)：
远程托管代码的服务器

git工作流程：
在工作目录添加修改文件 UserMapper.xml
将需要进行版本管理的文件放入缓存区 git add .
将缓存区的文件提交到git仓库 git commit 
因此，git管理过的文件三种状态（modified，staged，committed）