import asyncio

from backend.app.operations.manager import OperationManager
from backend.app.operations.schemas import OperationState


def test_fallback_search_operation_succeeds():
    async def run():
        manager = OperationManager()
        status = await manager.start("search", {"query": "last-second clutch", "limit": 3})
        final_status = await manager.wait(status.operation_id)
        assert final_status.state == OperationState.SUCCEEDED
        assert final_status.progress == 1.0
        assert "fallback" in final_status.result
        assert final_status.result["query"] == "last-second clutch"

    asyncio.run(run())


def test_operation_can_be_canceled():
    async def run():
        manager = OperationManager()
        status = await manager.start("index", {"source": "fixtures"})
        await asyncio.sleep(0.01)
        canceled = await manager.cancel(status.operation_id)
        final_status = await manager.wait(status.operation_id)
        assert canceled.cancel_requested is True
        assert final_status.state == OperationState.CANCELED

    asyncio.run(run())
