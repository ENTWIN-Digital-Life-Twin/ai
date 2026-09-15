from fastapi import APIRouter, Depends

from app.api.dependencies import get_form_suggest_service, get_task_duration_service
from app.schemas.form_suggest import FormSuggestRequest, FormSuggestResponse
from app.schemas.planning import TaskDurationRequest, TaskDurationResponse
from app.services.form_suggest_service import FormSuggestService
from app.services.task_duration_service import TaskDurationService

router = APIRouter(tags=["planning"])


@router.post("/task-duration", response_model=TaskDurationResponse)
def task_duration(
    payload: TaskDurationRequest,
    service: TaskDurationService = Depends(get_task_duration_service),
) -> TaskDurationResponse:
    return service.predict(payload)


@router.post("/form-suggest", response_model=FormSuggestResponse)
def form_suggest(
    payload: FormSuggestRequest,
    service: FormSuggestService = Depends(get_form_suggest_service),
) -> FormSuggestResponse:
    return service.suggest(payload)
