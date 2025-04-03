"""
Storage module for VIGIL data persistence.
"""

from vigil.storage.data_store import FileStore, SQLiteStore, init_storage

__all__ = ['FileStore', 'SQLiteStore', 'init_storage']
