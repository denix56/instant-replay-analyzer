# Progress

Long-running work is represented as an operation:

- `operation_id`
- `operation_type`
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `progress_percent`
- `current_step`
- `current_item`
- `total_items`
- `completed_items`
- `failed_items`
- `skipped_items`
- `message`
- `errors`

Statuses:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

Events are append-only and persisted:

- `operation_id`
- `timestamp`
- `event_type`
- `step`
- `status`
- `progress_percent`
- `current_item`
- `message`
- `metadata`

The backend exposes operation lists, operation detail, SSE event streams, and cancellation endpoints. Cancellation stops at safe checkpoints; partially processed clips remain valid.
