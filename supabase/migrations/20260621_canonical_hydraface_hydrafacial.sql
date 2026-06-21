insert into public.services (
    business_slug,
    category,
    category_key,
    service_name,
    variant_name,
    description,
    aliases,
    booking_mode,
    requires_duration_choice,
    duration_minutes,
    fixed_duration_minutes,
    allowed_durations_minutes,
    price_pence,
    price_by_duration,
    is_active,
    sort_order,
    display_order
)
select
    'veronika',
    'facials',
    'facials',
    'Hydraface / Hydrafacial',
    null,
    'Hydraface / Hydrafacial facial treatment with 60-minute and 90-minute options.',
    array[
        'hydraface',
        'hydraface facial',
        'hydraface facial treatment',
        'hydrafacial',
        'hydra facial',
        'hydra facial treatment'
    ],
    'choose_duration',
    true,
    null,
    null,
    array[60, 90],
    null,
    '{"60":6000,"90":8000}',
    true,
    0,
    0
where not exists (
    select 1
      from public.services
     where business_slug = 'veronika'
       and lower(btrim(service_name)) = 'hydraface / hydrafacial'
);

update public.services
   set category = 'facials',
       category_key = 'facials',
       service_name = 'Hydraface / Hydrafacial',
       variant_name = null,
       description = 'Hydraface / Hydrafacial facial treatment with 60-minute and 90-minute options.',
       aliases = array[
            'hydraface',
            'hydraface facial',
            'hydraface facial treatment',
            'hydrafacial',
            'hydra facial',
            'hydra facial treatment'
       ],
       booking_mode = 'choose_duration',
       requires_duration_choice = true,
       duration_minutes = null,
       fixed_duration_minutes = null,
       allowed_durations_minutes = array[60, 90],
       price_pence = null,
       price_by_duration = '{"60":6000,"90":8000}',
       is_active = true,
       sort_order = coalesce(sort_order, display_order, 0),
       display_order = coalesce(display_order, sort_order, 0)
 where id = (
    select id
      from public.services
     where business_slug = 'veronika'
       and lower(btrim(service_name)) = 'hydraface / hydrafacial'
     order by coalesce(sort_order, display_order, 9999), id
     limit 1
);

update public.services
   set is_active = false,
       description = 'Merged into Hydraface / Hydrafacial.'
 where business_slug = 'veronika'
   and id <> (
        select id
          from public.services
         where business_slug = 'veronika'
           and lower(btrim(service_name)) = 'hydraface / hydrafacial'
         order by coalesce(sort_order, display_order, 9999), id
         limit 1
   )
   and (
        lower(btrim(service_name)) in (
            'hydraface / hydrafacial',
            'hydraface',
            'hydraface facial',
            'hydraface facial treatment',
            'hydrafacial',
            'hydra facial',
            'hydra facial treatment',
            'hydrafacial 90 minutes',
            '90 minute hydrafacial'
        )
        or exists (
            select 1
              from unnest(coalesce(aliases, array[]::text[])) as alias(value)
             where lower(btrim(value)) in (
                'hydraface',
                'hydraface facial',
                'hydraface facial treatment',
                'hydrafacial',
                'hydra facial',
                'hydra facial treatment',
                'hydrafacial 90 minutes',
                '90 minute hydrafacial'
             )
        )
   );

insert into public.services (
    business_slug,
    category,
    category_key,
    service_name,
    variant_name,
    description,
    aliases,
    booking_mode,
    requires_duration_choice,
    duration_minutes,
    fixed_duration_minutes,
    allowed_durations_minutes,
    price_pence,
    price_by_duration,
    is_active,
    sort_order,
    display_order
)
select
    'veronika',
    'massage',
    'massage',
    'Swedish Massage',
    null,
    'Swedish Massage with selectable 30, 60, 90, and 120-minute options.',
    array['swedish massage', 'swedish'],
    'choose_duration',
    true,
    null,
    null,
    array[30, 60, 90, 120],
    null,
    '{"30":3500,"60":5000,"90":7500,"120":9500}',
    true,
    0,
    0
where not exists (
    select 1
      from public.services
     where business_slug = 'veronika'
       and lower(btrim(service_name)) = 'swedish massage'
);

update public.services
   set category = 'massage',
       category_key = 'massage',
       service_name = 'Swedish Massage',
       variant_name = null,
       description = 'Swedish Massage with selectable 30, 60, 90, and 120-minute options.',
       aliases = array['swedish massage', 'swedish'],
       booking_mode = 'choose_duration',
       requires_duration_choice = true,
       duration_minutes = null,
       fixed_duration_minutes = null,
       allowed_durations_minutes = array[30, 60, 90, 120],
       price_pence = null,
       price_by_duration = '{"30":3500,"60":5000,"90":7500,"120":9500}',
       is_active = true,
       sort_order = coalesce(sort_order, display_order, 0),
       display_order = coalesce(display_order, sort_order, 0)
 where id = (
    select id
      from public.services
     where business_slug = 'veronika'
       and lower(btrim(service_name)) = 'swedish massage'
     order by coalesce(sort_order, display_order, 9999), id
     limit 1
);

update public.services
   set is_active = false,
       description = 'Merged into Swedish Massage.'
 where business_slug = 'veronika'
   and id <> (
        select id
          from public.services
         where business_slug = 'veronika'
           and lower(btrim(service_name)) = 'swedish massage'
         order by coalesce(sort_order, display_order, 9999), id
         limit 1
   )
   and (
        lower(btrim(service_name)) = 'swedish massage'
        or exists (
            select 1
              from unnest(coalesce(aliases, array[]::text[])) as alias(value)
             where lower(btrim(value)) in ('swedish massage', 'swedish')
        )
   );
