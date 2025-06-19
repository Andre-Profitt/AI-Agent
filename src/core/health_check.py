"""
Comprehensive health check system
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import aiohttp

from src.utils.logging import get_logger

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """Individual health check"""
    
    def __init__(
        self,
        name: str,
        check_fn: Callable,
        critical: bool = True,
        timeout: float = 5.0
    ):
        self.name = name
        self.check_fn = check_fn
        self.critical = critical
        self.timeout = timeout
        self.last_check: Optional[datetime] = None
        self.last_status: Optional[HealthStatus] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0
    
    async def execute(self) -> tuple[HealthStatus, Optional[str]]:
        """Execute the health check"""
        try:
            result = await asyncio.wait_for(
                self.check_fn(),
                timeout=self.timeout
            )
            
            self.last_check = datetime.now()
            self.consecutive_failures = 0
            
            if result:
                self.last_status = HealthStatus.HEALTHY
                self.last_error = None
                return HealthStatus.HEALTHY, None
            else:
                self.last_status = HealthStatus.UNHEALTHY
                self.last_error = "Check returned False"
                return HealthStatus.UNHEALTHY, "Check failed"
                
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            self.last_status = HealthStatus.UNHEALTHY
            self.last_error = "Timeout"
            return HealthStatus.UNHEALTHY, "Health check timeout"
            
        except Exception as e:
            self.consecutive_failures += 1
            self.last_status = HealthStatus.UNHEALTHY
            self.last_error = str(e)
            return HealthStatus.UNHEALTHY, str(e)

class HealthChecker:
    """System-wide health checker"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        self._http_server: Optional[aiohttp.web.Application] = None
        
    def register_check(
        self,
        name: str,
        check_fn: Callable,
        critical: bool = True,
        timeout: float = 5.0
    ):
        """Register a health check"""
        self.checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            critical=critical,
            timeout=timeout
        )
        logger.info(f"Registered health check: {name}")
    
    async def start(self, db_manager=None, integration_hub=None):
        """Start health check system"""
        self._running = True
        
        # Register default checks
        if db_manager:
            self.register_check(
                "database",
                db_manager.test_connection,
                critical=True
            )
        
        if integration_hub:
            self.register_check(
                "integration_hub",
                lambda: integration_hub.initialized,
                critical=True
            )
        
        # Register system checks
        self.register_check(
            "disk_space",
            self._check_disk_space,
            critical=False
        )
        
        self.register_check(
            "memory",
            self._check_memory,
            critical=False
        )
        
        # Start periodic checks
        self._check_task = asyncio.create_task(self._run_checks())
        
        # Start HTTP endpoint
        await self._start_http_server()
        
        logger.info("Health check system started")
    
    async def stop(self):
        """Stop health check system"""
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            await asyncio.gather(self._check_task, return_exceptions=True)
        
        if self._http_server:
            await self._http_server.cleanup()
        
        logger.info("Health check system stopped")
    
    async def _run_checks(self):
        """Run health checks periodically"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self.check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        
        # Run checks concurrently
        tasks = {
            name: asyncio.create_task(check.execute())
            for name, check in self.checks.items()
        }
        
        for name, task in tasks.items():
            try:
                status, error = await task
                results[name] = {
                    "status": status.value,
                    "error": error,
                    "last_check": self.checks[name].last_check.isoformat()
                    if self.checks[name].last_check else None,
                    "critical": self.checks[name].critical
                }
            except Exception as e:
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(e),
                    "critical": self.checks[name].critical
                }
        
        return results
    
    async def get_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        check_results = await self.check_all()
        
        # Determine overall status
        critical_checks = [
            r for name, r in check_results.items()
            if r.get('critical') and r['status'] != HealthStatus.HEALTHY.value
        ]
        
        non_critical_unhealthy = [
            r for name, r in check_results.items()
            if not r.get('critical') and r['status'] != HealthStatus.HEALTHY.value
        ]
        
        if critical_checks:
            overall_status = HealthStatus.UNHEALTHY
        elif non_critical_unhealthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": check_results,
            "summary": {
                "total_checks": len(check_results),
                "healthy": sum(
                    1 for r in check_results.values()
                    if r['status'] == HealthStatus.HEALTHY.value
                ),
                "unhealthy": sum(
                    1 for r in check_results.values()
                    if r['status'] == HealthStatus.UNHEALTHY.value
                ),
                "critical_failures": len(critical_checks)
            }
        }
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space"""
        import psutil
        
        disk = psutil.disk_usage('/')
        # Fail if less than 10% free
        return disk.percent < 90
    
    async def _check_memory(self) -> bool:
        """Check memory usage"""
        import psutil
        
        memory = psutil.virtual_memory()
        # Fail if less than 10% free
        return memory.percent < 90
    
    async def _start_http_server(self):
        """Start HTTP server for health checks"""
        app = aiohttp.web.Application()
        
        async def health_handler(request):
            """Health check endpoint"""
            status = await self.get_status()
            
            # Determine HTTP status code
            if status['status'] == HealthStatus.HEALTHY.value:
                http_status = 200
            elif status['status'] == HealthStatus.DEGRADED.value:
                http_status = 200  # Still return 200 for degraded
            else:
                http_status = 503
            
            return aiohttp.web.json_response(status, status=http_status)
        
        async def ready_handler(request):
            """Readiness check endpoint"""
            # Simple check if system is ready
            ready = all(
                check.last_status == HealthStatus.HEALTHY
                for name, check in self.checks.items()
                if check.critical
            )
            
            if ready:
                return aiohttp.web.json_response({"ready": True})
            else:
                return aiohttp.web.json_response(
                    {"ready": False},
                    status=503
                )
        
        app.router.add_get('/health', health_handler)
        app.router.add_get('/ready', ready_handler)
        
        # Start server on separate port
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        
        site = aiohttp.web.TCPSite(runner, '0.0.0.0', 8001)
        await site.start()
        
        self._http_server = app
        logger.info("Health check HTTP server started on port 8001") 