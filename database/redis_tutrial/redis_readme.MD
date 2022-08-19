
## 参考链接:
[1] https://www.bilibili.com/video/BV1cr4y1671t?p=4&vd_source=7111d4cfa9354342c253c06ecdd64e2f
[2] https://www.bilibili.com/video/BV1KS4y1G7jm?spm_id_from=333.337.search-card.all.click



## Redis  
+ remote dictionary server  远程词典服务器

### 基本概念 

+ 是一个 key-value 的NoSQL非关系型数据库
+ 优点:
  + 基于内存操作, 读写速度非常快(基于内存、IO多路复用、良好的编码)
  + 支持丰富的数据结构
  + 支持事务, 操作都是原子性
  + 按key设置过期时间, 到期后自动删除
  + 支持主从 master-slave 集群复制来实现数据备份, 主机会自动将数据同步到从机
  + 单线程
  + 支持数据持久化
  + 支持多语言客户端

+ 缺点:
  + 内存数据库, 对机器内存要求
  + 如果要进行完整重同步, 由于需要生成rdb文件, 并进行传输, 会占用主机的CPU, 并消耗现网的带宽
  + 修改配置文件, 进行重启, 将硬盘中的数据加载进内存的耗时比较久

### 安装说明

参考链接:   

https://www.bilibili.com/video/BV1cr4y1671t?p=5&spm_id_from=pageDriver&vd_source=7111d4cfa9354342c253c06ecdd64e2f


### 数据结构

#### 基本类型
+ String
+ Hash
+ List
+ Set
+ SortedSet

#### 特殊类型
+ GEO
+ BitMap
+ HyperLog


### Redis命令

#### 通用命令
+ keys  查看符合模板的所有key, keys *
+ del 删除一个/多个key
+ exits 判断key是否存在
+ expire 给一个key设置有效期, 有效期到期时该key会被自动删除
+ ttl  查看一个key的剩余有效期


#### string类型命令
string类型, 即字符串类型, 是redis中最简单的存储类型, value是字符串, 分为3类: string 普通字符串 int 整数 float 浮点类型
  
string结构是将对象序列化为JSON字符串后存储, 当需要修改对象某个字段时很不方便.

+ set  添加或修改已经存在的一个string类型的键值对
+ get 根据key获取string类型的value
+ mset 批量添加多个string类型的键值对
+ mget 根据多个key获取多个string类型的value
+ incr  让一个整型的key自增1
+ incrby  让一个整型的key自增并指定步长
+ incrbyloat  让一个浮点类型的数字自增并指定步长
+ setnx  添加一个string类型的键值对, 前提是这个key不存在, 否则不执行
+ setex  添加一个string类型的键值对, 并且指定有效期


#### redis的key的格式
[项目名]:[业务名]:[类型]:[id]


#### hash类型
hash结构可以将对象中的每个字段独立存储, 可以针对单个字段做CRUD

+ hset key field value  添加或修改hash类型key的field的值  --  HSET heima:user:3 name Lucy
+ hget key field  获取一个hash类型key的field的值
+ hmset  批量添加多个hash类型key的field的值
+ hmget  批量获取多个hash类型key的field的值
+ hgetall  获取一个hash类型的key中的所有field和value
+ hkeys  获取一个hash类型的key中的所有的field
+ hvals  获取一个hash类型的key中的所有的value
+ hincrby  让一个hash类型key的字段值自增并指定步长
+ hsetnx  添加一个hash类型的key的field值, 前提是这个field不存在, 否则不执行


#### List类型
redis中的list类型与java中的林科大list类似, 可以看作是一个双向链表结构, 可以支持正向检索和反向检索.

特征: 有序 元素可以重复  插入和删除快 查询速度一般

+ lpush key element  向列表左边插入一个或多个元素
+ lpop key  移除并返回列表左侧的第一个元素, 没有则返回Null
+ rpush key element  向列表右侧插入一个或多个元素
+ lrange key star end  返回一段角标范围内的所有元素
+ blpop brpop  与lpop和rpop类似, 只不过在没有元素时等待指定时间, 而不是直接返回null


#### Set类型
与java中的hashset类似, 可以看作是一个value为null的hashmap

+ SADD key member  向set中添加一个或多个元素
+ srem key member  移除set中的指定元素
+ scard key  返回set中元素的个数
+ sismember key member  判断一个元素是否存在于member中


#### SortedSet类型
是一个可排序的set集合, 与java中的Treeset类似, 但是底层数据结构差别巨大.  
sortedset中的每一个元素都带有一个score属性, 可以基于score属性对元素排序, 底层的实现是一个跳表加hash表

+ zadd key score member  添加一个或多个元素到sorted set, 如果已经存在则更新其score值
+ zrem key member  删除sorted set 中的一个指定元素
+ zscore key member  获取sorted set 中的指定元素的score值










