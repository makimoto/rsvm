//! Kernel cache implementation
//!
//! Provides LRU cache for kernel matrix values to avoid redundant computations
//! in SMO algorithm. Kernel matrices are symmetric, so we only cache K(i,j) where i <= j.

use lru::LruCache;
use std::num::NonZeroUsize;

/// Cache key for kernel values, normalized so that i <= j
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CacheKey {
    i: usize,
    j: usize,
}

impl CacheKey {
    /// Create a normalized cache key where i <= j
    fn new(i: usize, j: usize) -> Self {
        if i <= j {
            Self { i, j }
        } else {
            Self { i: j, j: i }
        }
    }
}

/// LRU cache for kernel matrix values
pub struct KernelCache {
    cache: LruCache<CacheKey, f64>,
    hits: u64,
    misses: u64,
}

impl KernelCache {
    /// Create a new kernel cache with specified capacity in number of entries
    pub fn new(capacity: usize) -> Self {
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            cache: LruCache::new(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Create a kernel cache with capacity based on memory size in bytes
    /// Assumes 8 bytes per f64 value + overhead
    pub fn with_memory_limit(memory_bytes: usize) -> Self {
        let capacity = (memory_bytes / 16).max(1); // 16 bytes per entry (key + value + overhead)
        Self::new(capacity)
    }

    /// Get a kernel value from cache
    pub fn get(&mut self, i: usize, j: usize) -> Option<f64> {
        let key = CacheKey::new(i, j);
        if let Some(&value) = self.cache.get(&key) {
            self.hits += 1;
            Some(value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Put a kernel value into cache
    pub fn put(&mut self, i: usize, j: usize, value: f64) {
        let key = CacheKey::new(i, j);
        self.cache.put(key, value);
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            capacity: self.cache.cap().get(),
            size: self.cache.len(),
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub capacity: usize,
    pub size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_normalization() {
        let key1 = CacheKey::new(1, 5);
        let key2 = CacheKey::new(5, 1);
        assert_eq!(key1, key2);
        assert_eq!(key1.i, 1);
        assert_eq!(key1.j, 5);
    }

    #[test]
    fn test_kernel_cache_basic() {
        let mut cache = KernelCache::new(3);

        // Cache miss
        assert_eq!(cache.get(0, 1), None);
        assert_eq!(cache.stats().misses, 1);

        // Put and get
        cache.put(0, 1, 5.0);
        assert_eq!(cache.get(0, 1), Some(5.0));
        assert_eq!(cache.stats().hits, 1);

        // Symmetric access
        assert_eq!(cache.get(1, 0), Some(5.0));
        assert_eq!(cache.stats().hits, 2);
    }

    #[test]
    fn test_kernel_cache_lru_eviction() {
        let mut cache = KernelCache::new(2);

        cache.put(0, 1, 1.0);
        cache.put(1, 2, 2.0);
        cache.put(2, 3, 3.0); // Should evict (0,1)

        assert_eq!(cache.get(0, 1), None); // Evicted
        assert_eq!(cache.get(1, 2), Some(2.0)); // Still there
        assert_eq!(cache.get(2, 3), Some(3.0)); // Still there
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = KernelCache::new(10);

        // No accesses yet
        assert_eq!(cache.hit_rate(), 0.0);

        // All misses
        cache.get(0, 1);
        cache.get(1, 2);
        assert_eq!(cache.hit_rate(), 0.0);

        // Add some data and hits
        cache.put(0, 1, 1.0);
        cache.get(0, 1); // Hit
        cache.get(0, 1); // Hit

        // 2 hits, 2 misses = 50%
        assert_eq!(cache.hit_rate(), 0.5);
    }

    #[test]
    fn test_cache_with_memory_limit() {
        let cache = KernelCache::with_memory_limit(1000);
        assert!(cache.cache.cap().get() > 0);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = KernelCache::new(10);
        cache.put(0, 1, 1.0);
        cache.get(0, 1);

        cache.clear();

        assert_eq!(cache.get(0, 1), None);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 1); // From the get after clear
    }
}
