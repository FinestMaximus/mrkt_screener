import streamlit as st
import os
import pickle
from datetime import datetime, timedelta
import hashlib
from utils.logger import info, debug, warning, error  # Updated import for logging


class CacheManager:
    """Manager for caching data to reduce API calls"""

    def __init__(self, cache_dir="./cache", max_age_days=1):
        """Initialize the cache manager"""
        self.cache_dir = cache_dir
        self.max_age = timedelta(days=max_age_days)

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            info(f"Cache directory created at: {cache_dir}")

    def _get_cache_path(self, key):
        """Get the file path for a cache key"""
        # Create a hash of the key to use as filename
        key_hash = hashlib.md5(str(key).encode()).hexdigest()
        debug(f"Cache path for key '{key}': {key_hash}.pickle")
        return os.path.join(self.cache_dir, f"{key_hash}.pickle")

    def get(self, key):
        """Get data from cache if it exists and is not expired"""
        cache_path = self._get_cache_path(key)
        debug(f"Attempting to get cache for key: {key}")

        try:
            if os.path.exists(cache_path):
                # Check if cache is expired
                modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - modified_time > self.max_age:
                    info(f"Cache expired for key: {key}")
                    return None

                # Load from cache
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    debug(f"Data loaded from cache for key: {key}")
                    return data
            info(f"No cache found for key: {key}")
            return None
        except Exception as e:
            warning(f"Error reading from cache for key '{key}': {e}")
            return None

    def set(self, key, data):
        """Save data to cache"""
        if data is None:
            warning(f"No data to cache for key: {key}")
            return False

        cache_path = self._get_cache_path(key)
        debug(f"Saving data to cache for key: {key}")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            info(f"Data cached successfully for key: {key}")
            return True
        except Exception as e:
            warning(f"Error writing to cache for key '{key}': {e}")
            return False

    def clear(self, key=None):
        """Clear specific cache item or all cache"""
        try:
            if key:
                cache_path = self._get_cache_path(key)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    info(f"Cache cleared for key: {key}")
            else:
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith(".pickle"):
                        os.remove(file_path)
                        info(f"Cleared cache file: {filename}")
            return True
        except Exception as e:
            error(f"Error clearing cache: {e}")
            return False


# Decorated versions of st.cache functions for improved logging
def cached_data(func=None, ttl=3600, show_spinner="Loading data..."):
    """Enhanced wrapper around st.cache_data with logging"""

    def decorator(function):
        cached_func = st.cache_data(ttl=ttl, show_spinner=show_spinner)(function)

        def wrapper(*args, **kwargs):
            debug(f"Calling cached function: {function.__name__}")
            try:
                result = cached_func(*args, **kwargs)
                debug(f"Function {function.__name__} executed successfully")
                return result
            except Exception as e:
                error(f"Error in cached function {function.__name__}: {e}")
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
