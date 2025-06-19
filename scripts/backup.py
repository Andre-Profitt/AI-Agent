"""
Backup and Disaster Recovery System for GAIA
Comprehensive backup solution with multiple storage backends
"""

import os
import sys
import json
import shutil
import tarfile
import gzip
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import aiofiles
import boto3
from botocore.exceptions import ClientError
import schedule
import time
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Types of backups"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class StorageBackend(Enum):
    """Storage backends for backups"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_type: BackupType = BackupType.FULL
    storage_backend: StorageBackend = StorageBackend.LOCAL
    retention_days: int = 30
    compression: bool = True
    encryption: bool = True
    verify_backup: bool = True
    parallel_uploads: int = 4
    
    # Storage-specific config
    local_path: str = "./backups"
    s3_bucket: str = "gaia-backups"
    s3_prefix: str = "backups"
    gcs_bucket: str = "gaia-backups"
    azure_container: str = "gaia-backups"

@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    size_bytes: int
    checksum: str
    components: List[str]
    dependencies: List[str]
    status: str
    error_message: Optional[str] = None

class BackupManager:
    """Main backup manager"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_history: List[BackupMetadata] = []
        self.storage_backend = self._create_storage_backend()
        
        # Ensure backup directory exists
        if config.storage_backend == StorageBackend.LOCAL:
            Path(config.local_path).mkdir(parents=True, exist_ok=True)
    
    def _create_storage_backend(self):
        """Create appropriate storage backend"""
        if self.config.storage_backend == StorageBackend.S3:
            return S3BackupStorage(self.config)
        elif self.config.storage_backend == StorageBackend.GCS:
            return GCSBackupStorage(self.config)
        elif self.config.storage_backend == StorageBackend.AZURE:
            return AzureBackupStorage(self.config)
        else:
            return LocalBackupStorage(self.config)
    
    async def create_backup(self) -> BackupMetadata:
        """Create a complete backup of the GAIA system"""
        backup_id = self._generate_backup_id()
        timestamp = datetime.utcnow()
        
        logger.info(f"Starting backup: {backup_id}")
        
        try:
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type=self.config.backup_type,
                size_bytes=0,
                checksum="",
                components=[],
                dependencies=[],
                status="in_progress"
            )
            
            # Collect backup components
            components = await self._collect_backup_components()
            metadata.components = components
            
            # Create backup archive
            backup_path = await self._create_backup_archive(backup_id, components)
            
            # Calculate size and checksum
            file_size = os.path.getsize(backup_path)
            checksum = await self._calculate_checksum(backup_path)
            
            metadata.size_bytes = file_size
            metadata.checksum = checksum
            
            # Upload to storage backend
            await self.storage_backend.upload_backup(backup_path, backup_id)
            
            # Verify backup if enabled
            if self.config.verify_backup:
                await self._verify_backup(backup_id, checksum)
            
            # Clean up local file
            os.remove(backup_path)
            
            metadata.status = "completed"
            self.backup_history.append(metadata)
            
            logger.info(f"Backup completed: {backup_id} ({file_size} bytes)")
            return metadata
            
        except Exception as e:
            logger.error(f"Backup failed: {backup_id} - {e}")
            metadata.status = "failed"
            metadata.error_message = str(e)
            self.backup_history.append(metadata)
            raise
    
    async def _collect_backup_components(self) -> List[str]:
        """Collect all components to backup"""
        components = []
        
        # Vector store data
        vector_store_paths = [
            "./chroma_db",
            "./data/vector_store",
            "./data/embeddings"
        ]
        
        for path in vector_store_paths:
            if os.path.exists(path):
                components.append(path)
        
        # Memory system data
        memory_paths = [
            "./data/agent_memories",
            "./data/working_memory",
            "./data/episodic_memory"
        ]
        
        for path in memory_paths:
            if os.path.exists(path):
                components.append(path)
        
        # Tool learning data
        tool_paths = [
            "./data/tool_learning",
            "./data/tool_reliability",
            "./data/tool_preferences"
        ]
        
        for path in tool_paths:
            if os.path.exists(path):
                components.append(path)
        
        # Configuration files
        config_paths = [
            "./.env",
            "./config.json",
            "./settings.yaml"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                components.append(path)
        
        # Logs (last 7 days)
        log_paths = [
            "./logs",
            "./agent_fsm.log"
        ]
        
        for path in log_paths:
            if os.path.exists(path):
                components.append(path)
        
        # Database dumps (if applicable)
        db_paths = [
            "./data/database",
            "./data/supabase"
        ]
        
        for path in db_paths:
            if os.path.exists(path):
                components.append(path)
        
        return components
    
    async def _create_backup_archive(self, backup_id: str, components: List[str]) -> str:
        """Create compressed backup archive"""
        backup_filename = f"gaia_backup_{backup_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        backup_path = os.path.join("/tmp", backup_filename)
        
        with tarfile.open(backup_path, "w:gz") as tar:
            for component in components:
                if os.path.exists(component):
                    tar.add(component, arcname=os.path.basename(component))
        
        return backup_path
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _verify_backup(self, backup_id: str, expected_checksum: str):
        """Verify backup integrity"""
        logger.info(f"Verifying backup: {backup_id}")
        
        # Download backup for verification
        local_path = await self.storage_backend.download_backup(backup_id)
        
        try:
            # Calculate checksum
            actual_checksum = await self._calculate_checksum(local_path)
            
            if actual_checksum != expected_checksum:
                raise ValueError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
            
            logger.info(f"Backup verification passed: {backup_id}")
            
        finally:
            # Clean up downloaded file
            if os.path.exists(local_path):
                os.remove(local_path)
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    async def restore_backup(self, backup_id: str, restore_path: str = "./restore") -> bool:
        """Restore from backup"""
        logger.info(f"Starting restore from backup: {backup_id}")
        
        try:
            # Download backup
            backup_path = await self.storage_backend.download_backup(backup_id)
            
            # Create restore directory
            Path(restore_path).mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(restore_path)
            
            # Clean up downloaded file
            os.remove(backup_path)
            
            logger.info(f"Restore completed: {backup_id} -> {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {backup_id} - {e}")
            return False
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List available backups"""
        return await self.storage_backend.list_backups()
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup"""
        return await self.storage_backend.delete_backup(backup_id)
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        logger.info("Cleaning up old backups")
        
        backups = await self.list_backups()
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)
        
        for backup in backups:
            if backup.timestamp < cutoff_date:
                logger.info(f"Deleting old backup: {backup.backup_id}")
                await self.delete_backup(backup.backup_id)

class LocalBackupStorage:
    """Local file system storage backend"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_path = Path(config.local_path)
        self.backup_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to local storage"""
        dest_path = self.backup_path / f"gaia_backup_{backup_id}.tar.gz"
        shutil.copy2(local_path, dest_path)
        logger.info(f"Backup uploaded to local storage: {dest_path}")
    
    async def download_backup(self, backup_id: str) -> str:
        """Download backup from local storage"""
        backup_file = self.backup_path / f"gaia_backup_{backup_id}.tar.gz"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")
        
        # Copy to temp location
        temp_path = f"/tmp/gaia_restore_{backup_id}.tar.gz"
        shutil.copy2(backup_file, temp_path)
        
        return temp_path
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List local backups"""
        backups = []
        
        for backup_file in self.backup_path.glob("gaia_backup_*.tar.gz"):
            # Extract backup ID from filename
            backup_id = backup_file.stem.replace("gaia_backup_", "")
            
            # Get file stats
            stat = backup_file.stat()
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.fromtimestamp(stat.st_mtime),
                backup_type=BackupType.FULL,
                size_bytes=stat.st_size,
                checksum="",
                components=[],
                dependencies=[],
                status="completed"
            )
            
            backups.append(metadata)
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete local backup"""
        backup_file = self.backup_path / f"gaia_backup_{backup_id}.tar.gz"
        
        if backup_file.exists():
            backup_file.unlink()
            logger.info(f"Deleted local backup: {backup_id}")
            return True
        
        return False

class S3BackupStorage:
    """AWS S3 storage backend"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket = config.s3_bucket
        self.prefix = config.s3_prefix
    
    async def upload_backup(self, local_path: str, backup_id: str):
        """Upload backup to S3"""
        key = f"{self.prefix}/gaia_backup_{backup_id}.tar.gz"
        
        try:
            self.s3_client.upload_file(local_path, self.bucket, key)
            logger.info(f"Backup uploaded to S3: s3://{self.bucket}/{key}")
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def download_backup(self, backup_id: str) -> str:
        """Download backup from S3"""
        key = f"{self.prefix}/gaia_backup_{backup_id}.tar.gz"
        temp_path = f"/tmp/gaia_restore_{backup_id}.tar.gz"
        
        try:
            self.s3_client.download_file(self.bucket, key, temp_path)
            return temp_path
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    async def list_backups(self) -> List[BackupMetadata]:
        """List S3 backups"""
        backups = []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.tar.gz'):
                    # Extract backup ID from key
                    backup_id = key.split('/')[-1].replace('gaia_backup_', '').replace('.tar.gz', '')
                    
                    metadata = BackupMetadata(
                        backup_id=backup_id,
                        timestamp=obj['LastModified'],
                        backup_type=BackupType.FULL,
                        size_bytes=obj['Size'],
                        checksum="",
                        components=[],
                        dependencies=[],
                        status="completed"
                    )
                    
                    backups.append(metadata)
        
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete S3 backup"""
        key = f"{self.prefix}/gaia_backup_{backup_id}.tar.gz"
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted S3 backup: {backup_id}")
            return True
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False

class GCSBackupStorage:
    """Google Cloud Storage backend"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        # Implementation would use google-cloud-storage library
        pass

class AzureBackupStorage:
    """Azure Blob Storage backend"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        # Implementation would use azure-storage-blob library
        pass

class BackupScheduler:
    """Backup scheduler for automated backups"""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.scheduler = schedule.Scheduler()
    
    def schedule_daily_backup(self, time: str = "02:00"):
        """Schedule daily backup"""
        self.scheduler.every().day.at(time).do(self._run_backup)
        logger.info(f"Scheduled daily backup at {time}")
    
    def schedule_weekly_backup(self, day: str = "sunday", time: str = "03:00"):
        """Schedule weekly backup"""
        getattr(self.scheduler.every(), day).at(time).do(self._run_backup)
        logger.info(f"Scheduled weekly backup on {day} at {time}")
    
    def _run_backup(self):
        """Run backup job"""
        try:
            asyncio.run(self.backup_manager.create_backup())
            logger.info("Scheduled backup completed successfully")
        except Exception as e:
            logger.error(f"Scheduled backup failed: {e}")
    
    def start(self):
        """Start the scheduler"""
        logger.info("Starting backup scheduler")
        while True:
            self.scheduler.run_pending()
            time.sleep(60)

# Command line interface
async def main():
    """Main backup script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GAIA Backup System")
    parser.add_argument("--action", choices=["backup", "restore", "list", "cleanup"], 
                       default="backup", help="Action to perform")
    parser.add_argument("--backup-id", help="Backup ID for restore/delete")
    parser.add_argument("--config", default="./backup_config.json", help="Backup configuration file")
    parser.add_argument("--schedule", action="store_true", help="Start backup scheduler")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        config = BackupConfig(**config_data)
    else:
        config = BackupConfig()
    
    # Create backup manager
    backup_manager = BackupManager(config)
    
    if args.action == "backup":
        metadata = await backup_manager.create_backup()
        print(f"Backup completed: {metadata.backup_id}")
        
    elif args.action == "restore":
        if not args.backup_id:
            print("Error: --backup-id required for restore")
            return
        
        success = await backup_manager.restore_backup(args.backup_id)
        if success:
            print(f"Restore completed: {args.backup_id}")
        else:
            print(f"Restore failed: {args.backup_id}")
    
    elif args.action == "list":
        backups = await backup_manager.list_backups()
        print(f"Found {len(backups)} backups:")
        for backup in backups:
            print(f"  {backup.backup_id} - {backup.timestamp} - {backup.size_bytes} bytes")
    
    elif args.action == "cleanup":
        await backup_manager.cleanup_old_backups()
        print("Cleanup completed")
    
    if args.schedule:
        scheduler = BackupScheduler(backup_manager)
        scheduler.schedule_daily_backup()
        scheduler.start()

if __name__ == "__main__":
    asyncio.run(main()) 