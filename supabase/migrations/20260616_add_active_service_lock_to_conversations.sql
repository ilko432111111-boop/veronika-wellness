alter table public.conversations
add column if not exists active_service_id bigint,
add column if not exists active_service_name text,
add column if not exists active_service_source text;
