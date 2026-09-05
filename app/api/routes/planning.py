from fastapi import APIRouter, Depends

from app.api.dependencies import get_task_duration_service
from app.schemas.planning import TaskDurationRequest, TaskDurationResponse
from app.services.task_duration_service import TaskDurationService

router = APIRouter(tags=["planning"])


@router.post("/task-duration", response_model=TaskDurationResponse)
def task_duration(
    payload: TaskDurationRequest,
    service: TaskDurationService = Depends(get_task_duration_service),
) -> TaskDurationResponse:
    return service.predict(payload)
