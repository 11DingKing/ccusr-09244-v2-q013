from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Dict, List, Literal
from datetime import datetime, timezone


class OperationDataBase(BaseModel):
    robot_model_id: int = Field(..., description="机型ID")
    scene_id: int = Field(..., description="场景ID")
    skill_id: int = Field(..., description="技能ID")
    robot_serial: Optional[str] = Field(None, max_length=100, description="机器人序列号")

    motion_trajectory: Dict[str, Any] = Field(..., description="动作轨迹数据")
    perception_records: Dict[str, Any] = Field(..., description="感知记录数据")
    grasp_result: Optional[Dict[str, Any]] = Field(None, description="抓取结果")

    timestamp_start: datetime = Field(..., description="作业开始时间")
    timestamp_end: datetime = Field(..., description="作业结束时间")
    duration_ms: Optional[int] = Field(None, description="作业时长(毫秒)")

    environment_conditions: Optional[Dict[str, Any]] = Field(None, description="环境条件")
    hardware_status: Optional[Dict[str, Any]] = Field(None, description="硬件状态")

    @field_validator("timestamp_start", "timestamp_end")
    @classmethod
    def _normalize_to_naive_utc(cls, value: datetime) -> datetime:
        # SQLite 以 ISO 字符串保存时间，游标按键集字符串比较。统一存成朴素 UTC，
        # 避免同一时刻因带不带时区、时区偏移写法不同而产生不同的字符串表示。
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value


class OperationDataCreate(OperationDataBase):
    pass


class OperationDataUpdate(BaseModel):
    robot_model_id: Optional[int] = None
    scene_id: Optional[int] = None
    skill_id: Optional[int] = None
    robot_serial: Optional[str] = None
    motion_trajectory: Optional[Dict[str, Any]] = None
    perception_records: Optional[Dict[str, Any]] = None
    grasp_result: Optional[Dict[str, Any]] = None
    timestamp_start: Optional[datetime] = None
    timestamp_end: Optional[datetime] = None
    duration_ms: Optional[int] = None
    environment_conditions: Optional[Dict[str, Any]] = None
    hardware_status: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    completeness_score: Optional[float] = None
    data_grade: Optional[str] = Field(None, max_length=10)


class OperationDataResponse(BaseModel):
    id: int
    robot_model_id: int
    scene_id: int
    skill_id: int
    robot_serial: Optional[str] = None
    motion_trajectory: Dict[str, Any]
    perception_records: Dict[str, Any]
    grasp_result: Optional[Dict[str, Any]] = None
    timestamp_start: datetime
    timestamp_end: datetime
    duration_ms: Optional[int] = None
    environment_conditions: Optional[Dict[str, Any]] = None
    hardware_status: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    completeness_score: Optional[float] = None
    data_grade: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class OperationDataListResponse(BaseModel):
    total: int
    items: List[OperationDataResponse]
    page: int
    page_size: int
    # 游标模式相关字段；旧页码模式下保持为默认值，兼容既有客户端。
    mode: Literal["page", "cursor"] = "page"
    sort_order: Optional[Literal["asc", "desc"]] = None
    has_next: Optional[bool] = None
    next_cursor: Optional[str] = Field(None, description="下一页游标；has_next 为 false 时为 null")


class BatchOperationResultItem(BaseModel):
    index: int
    success: bool
    data: Optional[OperationDataResponse] = None
    error: Optional[str] = None


class BatchOperationResponse(BaseModel):
    total: int
    success_count: int
    failure_count: int
    results: List[BatchOperationResultItem]


class AnnotationBase(BaseModel):
    operation_data_id: int = Field(..., description="作业数据ID")
    is_success: bool = Field(..., description="是否成功")
    failure_category: Optional[str] = Field(None, max_length=50, description="失败大类")
    failure_subcategory: Optional[str] = Field(None, max_length=100, description="失败小类")
    failure_description: Optional[str] = Field(None, description="失败详细描述")
    annotator: Optional[str] = Field(None, max_length=100, description="标注人")


class AnnotationCreate(AnnotationBase):
    pass


class AnnotationUpdate(BaseModel):
    is_success: Optional[bool] = None
    failure_category: Optional[str] = Field(None, max_length=50)
    failure_subcategory: Optional[str] = Field(None, max_length=100)
    failure_description: Optional[str] = None
    annotator: Optional[str] = Field(None, max_length=100)
    review_status: Optional[str] = Field(None, max_length=20)
    reviewer: Optional[str] = Field(None, max_length=100)
    review_notes: Optional[str] = None
    annotation_quality_score: Optional[float] = None


class AnnotationResponse(BaseModel):
    id: int
    operation_data_id: int
    is_success: bool
    failure_category: Optional[str] = None
    failure_subcategory: Optional[str] = None
    failure_description: Optional[str] = None
    annotator: Optional[str] = None
    annotation_time: Optional[datetime] = None
    review_status: Optional[str] = None
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None
    annotation_quality_score: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
