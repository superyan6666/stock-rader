import os
import time
import threading
from typing import Optional
import pandas as pd
from multiprocessing import Value, Lock as MPLock
from utils.logger import logger

try:
    import pyarrow.feather as feather
    FEATHER_AVAILABLE = True
except ImportError:
    FEATHER_AVAILABLE = False

class SharedDFCache:
    """基于内存映射的零拷贝DataFrame缓存，多进程可直接读取 (破解 IPC 瓶颈)"""
    def __init__(self, cache_dir: str = "/dev/shm/quantbot" if os.path.exists("/dev/shm") else ".quantbot_data/shm", ttl_hours: int = 12):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self._cleanup_stale()

    def _cleanup_stale(self):
        """清理超过 TTL 的过期缓存文件"""
        now = time.time()
        try:
            for fname in os.listdir(self.cache_dir):
                fpath = os.path.join(self.cache_dir, fname)
                if fname.endswith('.feather') and os.path.isfile(fpath):
                    if now - os.path.getmtime(fpath) > self.ttl_seconds:
                        os.remove(fpath)
                        logger.debug(f"🧹 清理过期缓存: {fname}")
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")

    def get_cache_size_mb(self) -> float:
        """监控当前缓存物理占用大小"""
        try:
            total = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if f.endswith('.feather')
            )
            return total / 1024 / 1024
        except Exception:
            return 0.0

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if not FEATHER_AVAILABLE: return None
        path = os.path.join(self.cache_dir, f"{key}.feather")
        if os.path.exists(path):
            try:
                return feather.read_feather(path, memory_map=True)
            except Exception as e:
                logger.debug(f"Cache read error for {key}: {e}")
        return None

    def set(self, key: str, df: pd.DataFrame):
        if not FEATHER_AVAILABLE: return
        path = os.path.join(self.cache_dir, f"{key}.feather")
        tmp = path + ".tmp"
        try:
            feather.write_feather(df, tmp)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug(f"Cache write error for {key}: {e}")

_SHARED_CACHE = SharedDFCache()

class TokenBucket:
    """普通的基于线程锁的令牌桶限速器"""
    def __init__(self, rate: float = 5.0, capacity: float = 10.0):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            else:
                wait = (tokens - self._tokens) / self.rate
                time.sleep(wait)
                self._tokens = 0.0
                return wait

class CrossProcessTokenBucket:
    """跨进程安全的共享内存令牌桶"""
    def __init__(self, rate: float = 4.0, capacity: float = 8.0):
        self.rate = rate
        self.capacity = capacity
        self._tokens = Value('d', capacity)
        self._last_refill = Value('d', time.monotonic())
        self._lock = MPLock()

    def acquire(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill.value
            self._tokens.value = min(
                self.capacity,
                self._tokens.value + elapsed * self.rate
            )
            self._last_refill.value = now
            
            if self._tokens.value >= tokens:
                self._tokens.value -= tokens
                return 0.0
            else:
                wait = (tokens - self._tokens.value) / self.rate
                time.sleep(wait)
                self._tokens.value = 0.0
                return wait

_API_LIMITER = CrossProcessTokenBucket(rate=4.0, capacity=8.0)
_WORKER_LOCAL_LIMITER: Optional[TokenBucket] = None

def worker_pool_initializer(rate_per_worker: float, capacity: float):
    global _WORKER_LOCAL_LIMITER
    _WORKER_LOCAL_LIMITER = TokenBucket(rate=rate_per_worker, capacity=capacity)
    logger.debug(f"Worker {os.getpid()} 限速器初始化: 分配速率 {rate_per_worker:.2f} req/s")
