参考链接: 

https://django-redis-chs.readthedocs.io/zh_CN/latest/
https://blog.csdn.net/jiandanokok/article/details/109426427


## django-redis 
django-redis是一个使得django支持redis cache/session 后端的全功能组件

## 安装
    
    pip install django-redis

## 使用配置
修改settings.py文件

        # Django的缓存配置
        CACHES = {
            "default": {
                "BACKEND": "django_redis.cache.RedisCache",
                "LOCATION": "redis://172.16.179.142:6379/9",
                "OPTIONS": {
                    "CLIENT_CLASS": "django_redis.client.DefaultClient",
                }
            }
        }
 
# 配置session存储
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"


## 


