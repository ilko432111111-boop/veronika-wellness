-- Global owner control for pausing automatic customer replies.

alter table public.chatbot_settings
    add column if not exists auto_reply_enabled boolean not null default true;

update public.chatbot_settings
set auto_reply_enabled = true
where auto_reply_enabled is null;
